import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce
from sklearn.metrics import roc_auc_score
from models.deit import deit_tiny_patch16_224
import pickle

def exists(val):
    return val is not None

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
    

class ViViT_CV(pl.LightningModule):
    def __init__(
        self,
        args,
        pool = 'cls',
        dim_head = 64,
        dropout = 0.5,
        emb_dropout = 0.
    ):
        super().__init__()
        self.channel = args.channel
        self.f1_threshold = args.f1_threshold

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.global_average_pool = pool == 'mean'

        # Pre-trained Image Encoder
        self.image_encoder = deit_tiny_patch16_224(pretrained=True)
        # d*num_chn (num_chn: number of image channels)
        self.dim = args.dim*args.channel

        if args.load_embryocv:
            # add additional dim for non-image embryocv features. (ex: class labels)
            self.dim += args.dim

        self.frame_info_embedding = nn.Sequential(
            nn.LayerNorm(args.frame_info_dim),
            nn.Linear(args.frame_info_dim, args.dim),
            nn.LayerNorm(args.dim)
        )

        if args.global_info_dim > 0:
            self.global_info_embedding = nn.Sequential(
                nn.LayerNorm(args.global_info_dim),
                nn.Linear(args.global_info_dim, self.dim),
                nn.LayerNorm(self.dim)
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, args.frame_size + 2, self.dim)) # [cls_token, frame_info, frame_tokens, global_token]
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, self.dim)) if not self.global_average_pool else None
        self.temporal_transformer = Transformer(self.dim, args.temporal_depth, args.heads, dim_head, self.dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(self.dim, 1)
        self.image_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.args = args

        self.train_pred = []
        self.train_label = []
        self.train_prob = []

        self.train_slide = []
        self.train_embryo = []

        self.val_pred = []
        self.val_label = []
        self.val_prob = []

        self.val_slide = []
        self.val_embryo = []

        self.test_pred = []
        self.test_label = []
        self.test_prob = []

        self.test_slide = []
        self.test_embryo = []
    
    def forward(self, video, frame_info=None, global_info=None):
        b, channel, f, H, W = video.shape
        # video_feature (b, f, (d*chn))
        x = self.encode_image_pre_channel(video)

        if frame_info != None:
            frame_info = frame_info.view(b,f,-1)
            frame_info_feature = self.frame_info_embedding(frame_info)
            # feature (b, f, (d*chn+d))
            x = torch.cat((frame_info_feature, x), dim = -1)

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)
            # feature (b, 1+f, (d*chn+d))
            x = torch.cat((temporal_cls_tokens, x), dim = 1)

        if global_info != None:
            global_info_feature = self.global_info_embedding(global_info).unsqueeze(1)
            # feature (b, 1+f+1, (d*chn+d))
            x = torch.cat((x, global_info_feature), dim = 1)
        
        if global_info != None:
            # [cls, frame, global]
            x += self.pos_embedding 
        else:
            # [cls, frame]
            x += self.pos_embedding[:,:-1,:]

        x = self.dropout(x)
        # attend across time
        x = self.temporal_transformer(x)
        # excise out temporal cls token or average pool
        x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')
        x = self.to_latent(x)

        return self.mlp_head(x)

    def encode_image_pre_channel(self,image):
        # input is (batch, C, frame, H, W)
        b, channel, frame, H, W = image.shape

        with torch.no_grad():
            features = []
            for c in range(channel):
                temp = image[:,c:c+1,...]
                temp = torch.cat([temp,temp,temp],dim=1).permute(0,2,1,3,4)
                temp = rearrange(temp, 'b f c n d -> (b f) c n d', b = b)
                temp = F.interpolate(temp, size=(224, 224), mode='nearest')
                temp = self.image_norm(temp)
                temp_feature = rearrange(self.image_encoder.forward_features(temp), '(b f) d -> b f d', b = b)
                features.append(temp_feature)

            features = torch.cat(features,dim=-1) 

            return features
    
    def training_step(self, batch, batch_idx):
        x, y = batch['video_input'],  batch['label']
        by = batch['binary_label']

        if self.args.load_embryocv:
            frag = batch['CV']['frag']
            stage_raw = batch['CV']['stage_raw']
            frame_info = torch.cat([frag, stage_raw],dim=-1)
            # average across different focal settings
            frame_info = frame_info.mean(dim=-2)
        else:
            frame_info = None

        if self.args.load_EHR or self.args.load_EHRcv:
            EHR = batch['EHR']
        else:
            EHR = None

        out = self.forward(x, frame_info, EHR)
        loss = F.huber_loss(out.flatten(), y.flatten(), delta=0.2)

        print(out)
        print(y)

        preds = out > self.f1_threshold 
        for i in range(len(y.flatten())):
            self.train_label.append(int(by.flatten()[i]))
            self.train_pred.append(int(preds.flatten()[i]))
            self.train_prob.append(float(out.flatten()[i]))

        self.log("train_loss", loss, on_step=False, on_epoch=True,sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['video_input'],  batch['label']
        by = batch['binary_label']

        if self.args.load_embryocv:
            frag = batch['CV']['frag']
            stage_raw = batch['CV']['stage_raw']
            frame_info = torch.cat([frag,stage_raw],dim=-1)
            frame_info = frame_info.mean(dim=-2)
        else:
            frame_info = None

        if self.args.load_EHR or self.args.load_EHRcv:
            EHR = batch['EHR']
        else:
            EHR = None

        out = self.forward(x,frame_info,EHR)
        loss = F.huber_loss(out.flatten(), y.flatten(), delta=0.2)

        print(out)
        print(y)

        preds = out > self.f1_threshold
        for i in range(len(y.flatten())):
            self.val_label.append(int(by.flatten()[i]))
            self.val_pred.append(int(preds.flatten()[i]))
            self.val_prob.append(float(out.flatten()[i]))
            self.val_embryo.append(batch['embryo_id'][i])
            self.val_slide.append(batch['slide_id'][i])

        self.log("val_loss", loss, on_step=False, on_epoch=True,sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch['video_input'],  batch['label']
        by = batch['binary_label']

        if self.args.load_embryocv:
            frag = batch['CV']['frag']
            stage_raw = batch['CV']['stage_raw']
            frame_info = torch.cat([frag, stage_raw],dim=-1)
            frame_info = frame_info.mean(dim=-2)
        else:
            frame_info = None

        if self.args.load_EHR or self.args.load_EHRcv:
            EHR = batch['EHR']
        else:
            EHR = None

        out = self.forward(x,frame_info,EHR)
        loss = F.huber_loss(out.flatten(), y.flatten(), delta=0.2)

        print(out)
        print(y)

        preds = out > self.f1_threshold
        for i in range(len(y.flatten())):
            self.test_label.append(int(by.flatten()[i]))
            self.test_pred.append(int(preds.flatten()[i]))
            self.test_prob.append(float(out.flatten()[i]))
            self.test_embryo.append(batch['embryo_id'][i])
            self.test_slide.append(batch['slide_id'][i])

        self.log("test_loss", loss, on_step=False, on_epoch=True,sync_dist=True)

        return loss

    def calculate_result(self,pred,label):
        TP = sum((label == 1 and pred == 1) for label, pred in zip(label, pred))
        TN = sum((label == 0 and pred == 0) for label, pred in zip(label, pred))
        FP = sum((label == 0 and pred == 1) for label, pred in zip(label, pred))
        FN = sum((label == 1 and pred == 0) for label, pred in zip(label,pred))

        # Calculating metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
        accuracy_class_0 = TN / (TN + FP) if (TN + FP) != 0 else 0
        accuracy_class_1 = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        return accuracy, accuracy_class_0, accuracy_class_1, f1_score 
        
    def training_epoch_end(self, outputs):
        accuracy, accuracy_class_0, accuracy_class_1, f1_score  = self.calculate_result(self.train_pred, self.train_label)

        min_val = min(self.train_prob)
        max_val = max(self.train_prob)
        range_val = max_val - min_val
        self.train_prob = [(x - min_val) / range_val for x in self.train_prob]

        auc = roc_auc_score(self.train_label, self.train_prob)

        self.log("train_accuracy", accuracy, sync_dist=True)
        self.log("train_accuracy_class_0", accuracy_class_0, sync_dist=True)
        self.log("train_accuracy_class_1", accuracy_class_1, sync_dist=True)
        self.log("train_f1_score", f1_score, sync_dist=True)
        self.log("train_auc", auc, sync_dist=True)

        self.train_label = []
        self.train_pred = []
        self.train_prob = []

    def validation_epoch_end(self, outputs):
        accuracy, accuracy_class_0, accuracy_class_1, f1_score = self.calculate_result(self.val_pred, self.val_label)

        min_val = min(self.val_prob)
        max_val = max(self.val_prob)
        range_val = max_val - min_val
        self.val_prob = [(x - min_val) / range_val for x in self.val_prob]

        auc = roc_auc_score(self.val_label, self.val_prob)

        prediction = [self.val_embryo, self.val_slide, self.val_prob, self.val_label]

        with open("prediction/" + self.args.name + '.pkl', 'wb') as file:
            pickle.dump(prediction, file)

        print("roc_auc_score", auc)

        self.log("val_accuracy", accuracy, sync_dist=True)
        self.log("val_accuracy_class_0", accuracy_class_0, sync_dist=True)
        self.log("val_accuracy_class_1", accuracy_class_1, sync_dist=True)
        self.log("val_f1_score", f1_score, sync_dist=True)
        self.log("val_auc", auc, sync_dist=True)

        self.val_label = []
        self.val_pred = []
        self.val_prob = []

    def test_epoch_end(self, outputs):
        accuracy, accuracy_class_0, accuracy_class_1, f1_score = self.calculate_result(self.test_pred, self.test_label)

        min_val = min(self.test_prob)
        max_val = max(self.test_prob)
        range_val = max_val - min_val
        self.test_prob = [(x - min_val) / range_val for x in self.test_prob]

        auc = roc_auc_score(self.test_label, self.test_prob)

        prediction = [self.test_embryo, self.test_slide, self.test_prob, self.test_label]

        with open("prediction/" + self.args.name + '.pkl', 'wb') as file:
            pickle.dump(prediction, file)

        print("roc_auc_score",auc)

        self.log("test_accuracy", accuracy, sync_dist=True)
        self.log("test_accuracy_class_0", accuracy_class_0, sync_dist=True)
        self.log("test_accuracy_class_1", accuracy_class_1, sync_dist=True)
        self.log("test_f1_score", f1_score, sync_dist=True)
        self.log("test_auc", auc, sync_dist=True)

        self.test_label = []
        self.test_pred = []
        self.test_prob = []
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        return {"optimizer": optimizer}
