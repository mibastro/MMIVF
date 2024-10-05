import os
import csv
import numpy as np
import pickle
import json
import argparse
from PIL import Image, ImageFile
import random
import cv2
import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import torch
from torch.utils.data import Sampler
from dataloader.preprocessing import fix_sample_image_paths, pad_np_array, transform_frame
import dataloader.preprocessing as preprocessing

"""
General Data Structure (EHR & Videos)
    - masad: Treatment ID. Each treatment ID may contain 1 or 2 slides depending on the number of embryos. Each slide_id contains up to 8 embryos.
    - slide_id: Slide containing multiple embryos. The number of embryos per slide can vary.
    - embryo_id: In the video folders, it is labeled as WELL01, ..., WELL0X. In EHR data, it is labeled as either "Well" or "Embryo ID". This code uses "Well". In most rows, the "Well" and "Embryo ID" numbers are identical, but there are a few exceptions.
    - F'#': Focal setting of the microscope. It's recommended to use at most 3 different F settings close to F0 (e.g., F0, F-13, F3 OR F0, F-15, F15). The current implementation uses only the 'F0' setting.

EmbryoCV Output Structure
    - stage_raw: A 13-dimensional probability vector.
    - stage_smooth: An integer representing the class index.
    - boxes: Bounding boxes of the zona region. These are used to crop images, making training for zona, pronuclei, and blastomere predictions easier.
    - zona: Segmentation mask in a compressed io.BytesIO format. Use mask.decode(zona) to retrieve a 500x500 numpy array.
    - pronuclei: List of (x, y) polygon coordinates of pronuclei. Length can vary from 0 to the number of detected points.
    - blastomeres: List of (x, y) polygon coordinates of blastomeres. Length can vary from 0 to 8.

EmbryoCV Output Structure After Processing
    - stage_raw: Same as above.
    - stage_smooth: Same as above.
    - boxes: Same format (lower, left, height, width).
    - zona: 500x500 numpy array (int8 values ranging from 0 to 4).
    - pronuclei: Added key ['mask']: A list of instance masks in 500x500 numpy array format (values of 0s and 1s). The number of instances per image varies, and it can be empty.
    - blastomeres: Added key ['mask']: A list of instance masks. The number of instances per image varies, and it can be empty.

EmbryoCV Output Batch Structure
    - output['frag']:  "XXX.jpg": float -> batch x frames x Fids x chn (chn = 1, representing fragmentation ratio).
    - output['stage_raw']:  "XXX.jpg": 13x1 np.array float32 -> batch x frames x Fids x chn (chn = 13, representing class probabilities).
    - output['stage_smooth']:  "XXX.jpg": int -> batch x frames x Fids x chn (chn = 1, representing class label as an integer).
    - output['boxes']:  "XXX.jpg": [lower, left, height, width] -> batch x frames x Fids x chn (chn = 4, representing bounding box coordinates).
    - output['pronuclei']: Same as blastomeres -> batch x frames x height x width (single channel image).
    - output['blastomeres']: Same as pronuclei -> batch x frames x height x width (single channel image).
        Type1: "XXX.jpg": List of (x, y) polygon coordinates for pronuclei.
        Type2: "XXX.jpg": Instance mask per instance (one image per instance).
        Type3: "XXX.jpg": Instance masks merged into one mask (use this format).
    - output['zona']:  "XXX.jpg": 500x500 numpy array (int8 values ranging from 0 to 4) -> batch x frames x height x width (single channel image).

Embryo IDs Structure
    - idx: Index of the row in the EHR data.
    - slide_id: Slide ID.
    - well: Well ID.
    - transferred: Total number of embryos transferred in this cycle (slide_id).
    - masad: Treatment ID (not used).
    - embryo_id: Embryo ID (not used).
    
    ### Following keys are added after filtering & matching processes:
    - video_path: Path to the video data.
    - embryocv_path: Path to the EmbryoCV output data.
"""

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

embryocv_output_tasks = ['frag', 'stage_raw', 'stage_smooth', 'boxes', 'pronuclei', 'blastomeres', 'zona']
EHR_field_id = ['SlideID', 'Well', 'Embryo ID', 'details-id', 'PatientNumber', 'PatinetCard_Patient_no'] # These fields are used to identify embryos or treatment. ignore 'Embryo ID'
EHR_field_embryocv = ['zona_width_mean', 'zona_width_mean_err', 'zona_width_std', 'zona_width_std_err', 'zona_inner_diameter_max', # EmbryoCV (interpretable) features
                      'zona_inner_diameter_max_err', 'zona_inner_diameter_min', 'zona_inner_diameter_min_err', 'zona_outer_diameter_max', 
                      'zona_outer_diameter_max_err', 'zona_outer_diameter_min', 'zona_outer_diameter_min_err', 
                      'frag_day2_median', 'frag_day2_iqr', 'frag_day3_median', 'frag_day3_iqr', '2-cell_time', '3-cell_time', '4-cell_time', 
                      '5-cell_time', '6-cell_time', '7-cell_time', '8-cell_time', '9+-cell_time', 'morula_time', 'blastocyst_time', 'empty_interval_0_starts_by', 
                      'empty_interval_0_ends_by', 'empty_interval_1_starts_by', 'empty_interval_1_ends_by', 'empty_interval_2_starts_by', 'empty_interval_2_ends_by',
                      'empty_interval_3_starts_by', 'empty_interval_3_ends_by', 'empty_interval_4_starts_by', 'empty_interval_4_ends_by', 'empty_interval_5_starts_by',
                      'empty_interval_5_ends_by', 'empty_interval_6_starts_by', 'empty_interval_6_ends_by', 'empty_interval_7_starts_by', 'empty_interval_7_ends_by', 
                      'start_movie', 'end_movie', 'zygote_area', 'zygote_area_err', 'zygote_shape', 'zygote_shape_err', '2-cell_symmetry', '4-cell_symmetry', 
                      'pn_appear_time', 'pn_fade_time', 'prob_0_pn', 'prob_1_pn', 'prob_2_pn', 'prob_3+_pn']
EHR_field_patient = [ 'patient_age', 'PatinetCard_BMI', 'AgeOfFirstMenstrual', 'TotalRetrievedOocytes', 'IVF_IvfType', 'e2-1', 'e2-2', 'e2-3']  # EHR features
# EHR_field_patient = [ 'patient_age', 'AgeOfFirstMenstrual', 'TotalRetrievedOocytes', 'e2-1', 'e2-2', 'e2-3']  # EHR features
EHR_field_outcome = ['children_N', 'tot_number_embryos_transferred'] # Results to predict (Ground truth)


'''
    __init__:
    Input:
        - args: Arguments passed from the main function.
        - split: Specifies the data split, which can be 'train', 'val', or 'test'.
        - modality: A list of modalities to load, such as ['video', 'EHR', 'embryocv', 'transferred'].
'''

class MultiModalLoader(Dataset):
    def __init__(self, 
                 args, 
                 split=None, 
                 modality=['video', 'EHR', 'embryocv', 'transferred'], 
                 augmentation=False,
                 train=False
                 ):
        self.embryocv_output_tasks = ['frag', 'stage_raw', 'stage_smooth', 'boxes', 'pronuclei', 'blastomeres', 'zona']
        self.args = args
        self.split = split
        self.modality = modality
        self.root = args.root
        self.root_EHR = args.root_EHR
        self.EHR_filename = args.EHR_filename
        self.root_videos = [os.path.join(args.root, args.video_dir)] # video paths. This is a list of paths.
        self.root_embryocv = os.path.join(args.root, args.embryocv_dir) # EmbryoCV output path
        self.embryocv_slide_ids_videos = [slide_id for slide_id in os.listdir(self.root_embryocv) if 'pdb' in slide_id]
        self.embryocv_slide_ids_unique = list(set(self.embryocv_slide_ids_videos))
        self.F = 'F0' # if you want to use multiple videos with different F (focal settings), getitem should be modified. (Not implemented)
        # all uses outputs from all F settings. EmbryoCV contains 3F settings. F0 and two other F settings that are closest to F0.
        self.F_per_task = {'frag': 'all', 
                           'stage_raw': 'all', 
                           'stage_smooth': 'all', 
                           'boxes': 'F0',       # 'all' not implemented
                           'pronuclei': 'F0',   # 'all' not implemented
                           'blastomeres': 'F0', # 'all' not implemented
                           'zona': 'F0'}        # 'all' not implemented
        self.max_frame = args.frame_size * args.fix_step
        self.frame_length =  args.frame_size
        self.crop_ROI = args.crop_ROI # whether to crop bbox region for pronuclei and blastomeres predictions
        self.corrupted_slides = []

        # load embryo ids csv. This is used to match EHR rows and videos.
        with open(os.path.join(self.root, args.matching_filename), 'r') as f:
            csv_data = csv.DictReader(f)
            data = [row for row in csv_data]
            embryo_ids = [{'row_idx': row[''], 'slide_id': row['SlideID'].replace('.','_'), 'well': row['Well'], 'selection': row['Selection'],
                                'masad': row['Treatment ID'], 'embryo_id': row['Embryo ID'], 'transferred': row['Transferred']}
                                for row in data]
                            
        # 1. filter the embryo_ids without video data
        if 'video' in self.modality:
            self.get_embryos_with_video(embryo_ids)
            embryo_ids = self.embryo_ids_w_video

        # 2. filter EHR data without embryoCV outputs
        if 'embryocv' in self.modality:
            self.get_embryos_with_embryocv_outputs(embryo_ids)
            embryo_ids = self.embryo_ids_w_embryocv

        # 3. filter embryo_ids that are not transferred to patients
        if 'transferred' in self.modality:
            self.get_embryos_transferred_only(embryo_ids)
            embryo_ids = self.embryo_ids_transferred

        # 4. get embryolist from predefined splits. split can be 'train', 'val', or 'test'
        with open(os.path.join(self.root, 'process_data', 'train_val_test.csv'), 'r') as f:
            csv_data = csv.DictReader(f)
            self.embryo_splits = [row for row in csv_data]
        self.get_embryos_from_split(embryo_ids)
        if split == 'train':
            embryo_ids = self.embryo_ids_in_train
        elif split == 'val':
            embryo_ids = self.embryo_ids_in_val
        elif split == 'test':
            embryo_ids = self.embryo_ids_in_test
        self.embryo_ids_all = self.embryo_ids_in_train + self.embryo_ids_in_val + self.embryo_ids_in_test
            

        # 5. load and preprocess the EHR data with the embryo_ids
        # save filtered EHR for faster the loading speed.
        if 'EHR' in self.modality:
            EHR_data = self.preprocess_EHR(
                output_column = 'children_N',
                embryo_ids = embryo_ids,
                split = split
            )
            # convert EHR_data to dictionary format
            self.EHR_data = {os.path.join(data['SlideID']+'_pdb', f"WELL{int(data['Well']):02}"):data for data in EHR_data}

            # 6. filter embryo_ids that does not have EHR data and label
            self.get_embryos_with_EHR(EHR_data, embryo_ids)
            embryo_ids = self.embryo_ids_w_EHR

        # Assign embryo_ids to the object instance variable
        self.embryo_ids = embryo_ids
        self.augmentation = augmentation
        self.batch_transform = BatchTransform(target_size=(324, 324), mean=[0.485], std=[0.229],augmentation=augmentation)
        self.train = train
      
    def __getitem__(self, idx):
        embryo_id = self.embryo_ids[idx]
        data = {}

        if self.augmentation:
            angle = random.uniform(-90, 90)  # Random angle between -90 and 90
            do_flip = random.random() > 0.5  # 50% chance to apply flip
        else:
            angle = 0
            do_flip = 0

        # 1. load video
        data['slide_id'] = embryo_id['slide_id']
        data['embryo_id'] = embryo_id['embryo_id']
        well_id = f'WELL{int(embryo_id["well"]):02}'
        video_dir = os.path.join(embryo_id['video_path'], embryo_id['slide_id'] + '_pdb', well_id, self.F)
        image_files = os.listdir(video_dir)
        image_files = sorted([f for f in image_files if f.endswith('.jpg')])
        image_files = fix_sample_image_paths(image_files, self.args.fix_step)
        embryo_id['num_frame'] = len(image_files)
        if 'video' in self.modality:
            video = self.read_and_preprocess_video(video_dir,image_files, do_flip, angle)    
            data['video'] = video
        
        # 2. load EHR
        slide_well_id =  os.path.join(embryo_id['slide_id'] + '_pdb', well_id)
        EHR_embryo = self.EHR_data[slide_well_id] # all EHR columns are retrieved.
        EHR_id = {key:val for key, val in EHR_embryo.items() if key in EHR_field_id}
        EHR_embryocv = {key:val for key, val in EHR_embryo.items() if key in EHR_field_embryocv}
        EHR_patient = {key:val for key, val in EHR_embryo.items() if key in EHR_field_patient}
        EHR_outcome = {key:val for key, val in EHR_embryo.items() if key in EHR_field_outcome}
        if EHR_outcome['tot_number_embryos_transferred'] == 0: # overwrite the transferred embryos when 0.
            EHR_outcome['tot_number_embryos_transferred'] = int(embryo_id['transferred'])
        data['label'] = torch.tensor(EHR_outcome['children_N'] / EHR_outcome['tot_number_embryos_transferred'])
        
        data["EHR"] = []
        if self.args.load_EHR:
            data['EHR'] += list(EHR_patient.values())
        if self.args.load_EHRcv:
            data['EHR'] +=  list(EHR_embryocv.values())

        # Replace Nan values with -1
        if data["EHR"] != []:
            data['EHR'] = [-1 if np.isnan(x) else x for x in data['EHR']]
            data['EHR'] = torch.Tensor(data["EHR"])
        else:
            del(data["EHR"])

        # 3. load embryovision results
        if 'embryocv' in self.modality:
            embryocv_path = embryo_id['embryocv_path']
            embryocv_output = self.load_embryovision_outputs(embryocv_path, well_id)
            if embryocv_output is not None:
                embryocv_output = self.process_embryovision_outputs(embryocv_output, Fid=self.F_per_task) # convert io.Bytes and xy_polygons to mask format.
            else:
                return self.__getitem__(idx+1)
            
            embryocv_parsed = self.parse_embryocv_output(embryocv_output, image_files, max_frame=self.max_frame, crop_ROI=self.crop_ROI)
            if embryocv_parsed is None:
                return self.__getitem__(idx+1)
            data['CV'] = embryocv_parsed 

            for temp in data['CV']:
                data['CV'][temp] = torch.Tensor(data['CV'][temp])

            data['CV']['pronuclei'] = data['CV']['pronuclei'][::self.args.fix_step]
            data['CV']['blastomeres'] = data['CV']['blastomeres'][::self.args.fix_step]
            data['CV']['zona'] = data['CV']['zona'][::self.args.fix_step]
            
            if self.augmentation:
                for i in range(data['CV']['pronuclei'].shape[0]):
                    data['CV']['pronuclei'][i:i+1] = transform_frame(data['CV']['pronuclei'][i:i+1], angle, do_flip)
                    data['CV']['blastomeres'][i:i+1] = transform_frame(data['CV']['blastomeres'][i:i+1], angle, do_flip)
                    data['CV']['zona'][i:i+1] = transform_frame(data['CV']['zona'][i:i+1], angle, do_flip)

            data['CV']['frag'] = data['CV']['frag'][::self.args.fix_step]
            data['CV']['stage_raw'] = data['CV']['stage_raw'][::self.args.fix_step]
        
        # 4. construct input tensor
        if 'video' in self.modality:
            if 'embryocv' in self.modality:
                data['video_input'] = torch.stack([data['video'].squeeze(0), data['CV']['pronuclei'], data['CV']['blastomeres'], data['CV']['zona']],dim=0)
            else:
                data['video_input'] = data['video'].unsqueeze(0) 
        else:
            if self.load_embryocv:
                data['video_input'] = torch.stack([data['CV']['pronuclei'], data['CV']['blastomeres'], data['CV']['zona']],dim=0)

        # 'binary label' is only used for evaluation (AUC calculation). 'label' is used for training.
        if data['label'] == 0:
            data['binary_label'] = torch.tensor(0)
        else:
            data['binary_label'] = torch.tensor(1)

        return data

    def __len__(self):  
        return len(self.embryo_ids)

    def __print__(self):
        # put the code that prints the arguments inside self.args
        for key, val in self.args.__dict__.items():
            print(f'{key}: {val}')

    def get_embryos_with_video(self, embryo_ids):
        self.embryo_ids_w_video = []
        self.embryo_ids_wo_video = []

        for embryo_id in embryo_ids:
            slide_id = embryo_id['slide_id'].replace('.', '_') + '_pdb'
            well_id = f'WELL{int(embryo_id["well"]):02}'

            for root_video in self.root_videos:
                if os.path.exists(os.path.join(root_video, slide_id, well_id, self.F)):
                    embryo_id['video_path'] = root_video
                    self.embryo_ids_w_video.append(embryo_id)
                    break
                elif root_video == self.root_videos[-1]:
                    embryo_id['video_path'] = None
                    self.embryo_ids_wo_video.append(embryo_id)
        
        print('\n----- filtering by video data -----')
        print('Total embryos in csv:', len(embryo_ids))
        print('Total embryos with video:', len(self.embryo_ids_w_video))
        print('Total embryos without video:', len(self.embryo_ids_wo_video))

    def get_embryos_with_embryocv_outputs(self, embryo_ids):
        self.embryo_ids_w_embryocv = []
        self.embryo_ids_wo_embryocv = []

        embryo_slide_ids = sorted(list(set([embryo_id['slide_id'] for embryo_id in embryo_ids])))
        
        for embryo_id in embryo_ids:
            slide_well_path = os.path.join(embryo_id['slide_id']+'_pdb', f"WELL{int(embryo_id['well']):02}")
            embryocv_paths = []
            if os.path.exists(os.path.join(self.root_embryocv, slide_well_path)):
                embryocv_paths.append(os.path.join(self.root_embryocv, slide_well_path))
            else:
                embryocv_paths = None
            embryo_id['embryocv_path'] = embryocv_paths

            if embryocv_paths is not None:
                self.embryo_ids_w_embryocv.append(embryo_id)
            else:
                self.embryo_ids_wo_embryocv.append(embryo_id)
        
        print('\n----- filtering by embryocv outputs -----')
        print('Total embryos before embryocv matching:', len(embryo_ids))
        print('Total embryos with embryocv:', len(self.embryo_ids_w_embryocv))
        print('Total embryos without embryocv:', len(self.embryo_ids_wo_embryocv))
        
        count_w_embryocv = 0
        count_wo_embryocv = 0
        for slide_id in embryo_slide_ids:
            if slide_id +"_pdb" in self.embryocv_slide_ids_unique:
                count_w_embryocv += 1
            else:
                count_wo_embryocv += 1

        print('Total slide_ids before embryocv matching:', len(embryo_slide_ids))
        print('Total slide_ids with embryocv:', count_w_embryocv)
        print('Total slide_ids without embryocv:', count_wo_embryocv)

    def get_embryos_transferred_only(self, embryo_ids):
        self.embryo_ids_transferred = []
        self.embryo_ids_nontransferred = []

        for embryo_id in embryo_ids:
            if embryo_id['selection'] == 'Transfer':
                self.embryo_ids_transferred.append(embryo_id)
            else:
                self.embryo_ids_nontransferred.append(embryo_id)
        
        print('----- filtering by transferred embryos -----')
        print('Total embryos in this loader:', len(embryo_ids))
        print('Total transferred embryos:', len(self.embryo_ids_transferred))
        print('Total non-transferred embryos:', len(self.embryo_ids_nontransferred))
    
    def get_embryos_from_split(self, embryo_ids):
        self.embryo_ids_in_train = []
        self.embryo_ids_in_val = []
        self.embryo_ids_in_test = []
        for embryo_id in embryo_ids:
            slide_id = embryo_id['slide_id'] + '_pdb' if 'pdb' not in embryo_id['slide_id'] else embryo_id['slide_id']
            well_id = f"WELL{int(embryo_id['well']):02}" if 'WELL' not in embryo_id['well'] else embryo_id['well']
            for id_split in self.embryo_splits:
                if (slide_id == id_split['slide_ids']) and (well_id in eval(id_split['well_ids'])):
                    if id_split['split'] == 'train':
                        self.embryo_ids_in_train.append(embryo_id)
                        break
                    elif id_split['split'] == 'val':
                        self.embryo_ids_in_val.append(embryo_id)
                        break
                    elif id_split['split'] == 'test':
                        self.embryo_ids_in_test.append(embryo_id)
                        break
            
        print('----- filtering by split -----')
        print(f'Total embryos in this loader:', len(embryo_ids))
        print(f'Total embryos in train split:', len(self.embryo_ids_in_train))
        print(f'Total embryos in val split:', len(self.embryo_ids_in_val))
        print(f'Total embryos in test split:', len(self.embryo_ids_in_test))

    def get_embryos_with_EHR(self, EHR_data, embryo_ids):
        EHR_data_embryoIDs = [os.path.join(row['SlideID'].replace('.', '_')+'_pdb', f"WELL{int(row['Well']):02}") for row in EHR_data]
        self.embryo_ids_w_EHR = []
        self.embryo_ids_wo_EHR = []

        for embryo_id in embryo_ids:
            slide_well_id = os.path.join(embryo_id['slide_id'].replace('.', '_')+'_pdb', f"WELL{int(embryo_id['well']):02}")
            if slide_well_id in EHR_data_embryoIDs:
                self.embryo_ids_w_EHR.append(embryo_id)
            else:
                self.embryo_ids_wo_EHR.append(embryo_id)
        
        print('\n----- filtering by EHR data -----')
        print('Total embryos in this loader:', len(embryo_ids))
        print('Total embryos with EHR:', len(self.embryo_ids_w_EHR))
        print('Total embryos without EHR:', len(self.embryo_ids_wo_EHR))

    def load_and_match_EHR(self, embryo_ids=None, split=None):
        if os.path.exists(os.path.join(self.root_EHR, f'{split}_'+self.EHR_filename)):
            df = pd.read_csv(os.path.join(self.root_EHR, f'{split}_'+self.EHR_filename))
        else:
            df = pd.read_csv(os.path.join(self.root_EHR, self.EHR_filename))
            df_filtered = df[df.apply(lambda row: any((row['SlideID'] == embryo_id['slide_id']) and (row['Well'] == int(embryo_id['well'])) for embryo_id in embryo_ids), axis=1)]
            df_filtered.to_csv(os.path.join(self.root_EHR, f'{split}_'+self.EHR_filename), index=False)
            print(f'the following file is filtered to match {split} samples: {os.path.join(self.root_EHR, split+self.EHR_filename)}')
            df = df_filtered
        return df

    def preprocess_EHR(self, output_column, embryo_ids=None, split=None):
        if split in ['train', 'val', 'test', 'transferred', 'entire']:
            df = self.load_and_match_EHR(embryo_ids, split)
        else:
            assert 0, 'check split name'
      
        # Visual features
        visual_features_range = df.columns.slice_locs('SlideID', 'Embryo ID')
        visual_features = []
        for i in range(visual_features_range[0], visual_features_range[1]):
            visual_features.append(df.columns[i])

        # Output features
        result_features = [
            'children_N', 
            'bHcg_max', 
            'pulse_N', 
            'bag_N'
        ]

        # For debugging purposes, maintain a dictionary with why each column was dropped
        dropReasons = {}

        # Filtering out non-numerical features
        df, mp_tmp = preprocessing.filterNumerical(df, keepCols = result_features + ['SlideID'])
        dropReasons.update(mp_tmp)

        # Filtering out columns by nan thresh
        df, mp_tmp = preprocessing.filterFew(df, nan_thresh = 0.4, keepCols = result_features + visual_features)
        dropReasons.update(mp_tmp)
        
        # Filtering out features measured after pregnancy 
        df, mp_tmp = preprocessing.removePostFeatures(df, keepCols = [output_column] + visual_features)
        dropReasons.update(mp_tmp)

        # Filter by correlation against output_column
        df, mp_tmp = preprocessing.filterCorrelation(df, result_i = output_column, keepCols = visual_features, corr_thresh = 0.05)
        dropReasons.update(mp_tmp)

        # Convert to list format
        return df.to_dict(orient='records')
 
    def read_and_preprocess_video(self,
                                  video_dir,
                                  image_paths,
                                  flip,
                                  angle,
                                  target_size=(328, 328)):
        # Load and preprocess images
        images= []
        for path in image_paths:
            try:
                img = Image.open(video_dir + "/" + path)
                images.append(img)
            except OSError as e:
                print(f"Error loading image {path}: {e}")

        tensors = self.batch_transform(images,flip,angle)

        # Pad the sequence with zeros if it's shorter than the target number of frames
        if len(tensors) < self.frame_length:
            padding = [torch.zeros(1, *target_size) for _ in range(self.frame_length - len(tensors))]
            tensors.extend(padding)
        # Truncate the sequence if it's longer than the target number of frames
        tensors = tensors[:self.frame_length] 
        # Stack all tensors into a single tensor
        video_tensor = torch.stack(tensors,dim=1)

        return video_tensor

    def load_embryovision_outputs(self, embryocv_paths, well_id):
        data_exists = False
        embryocv_output = {}
        for embryocv_path in embryocv_paths:
            # load the embryocv outputs
            slide_id = embryocv_path.split('/')[-2]
            with open(os.path.join(embryocv_path, 'frag.pkl'), 'rb') as f:
                frag = pickle.load(f)
                well_ids_in_data = frag[slide_id].keys()
                if well_id not in well_ids_in_data:
                    continue
                else:
                    data_exists = True
                    embryocv_output['frag'] = frag[slide_id][well_id]
            
            for task in self.embryocv_output_tasks[1:]:
                with open(os.path.join(embryocv_path, f'{task}.pkl'), 'rb') as f:
                    try:
                        task_output = pickle.load(f)
                    except:
                        embryocv_output[task] = []
                        continue
                    
                    if (slide_id not in task_output.keys()) or (well_id not in task_output[slide_id].keys()):
                        embryocv_output[task] = []
                    else:
                        embryocv_output[task] = task_output[slide_id][well_id]

        return embryocv_output if data_exists else None

    def process_embryovision_outputs(self, embryocv_output, Fid='F0', skip_process=False):
        
        if isinstance(Fid, str):
            Fid_dict = {task:Fid for task in self.embryocv_output_tasks}
        elif isinstance(Fid, dict):
            Fid_dict = dict(Fid)

        task_not_exist = []
        for task in ['frag', 'stage_raw', 'stage_smooth', 'boxes']:
            Fid = Fid_dict[task]
            if (task in self.embryocv_output_tasks) and (embryocv_output[task] is not None) and ('F0' in embryocv_output[task].keys()) and (len(embryocv_output[task].keys())==3):
                embryocv_output[task] = embryocv_output[task]
            else:
                embryocv_output[task] = []
                task_not_exist.append(task)
        # list of (xy polygon coordinates format)
        if ('pronuclei' in self.embryocv_output_tasks) and (embryocv_output['pronuclei'] is not None) and (Fid in embryocv_output['pronuclei'].keys()):
            pronuclei = embryocv_output['pronuclei'][Fid]
            if not skip_process:
                for image_id, cells in pronuclei.items():
                    if len(cells) == 0:
                        continue
                    for cell in cells:
                        segm = np.zeros([500,500], np.uint8)
                        cell['mask'] = cv2.fillPoly(segm, np.array([cell['xy_polygon']], dtype=np.int32), 1)
        else:
            pronuclei = []
            task_not_exist.append('pronuclei')
        # list of (xy polygon coordinates format)
        if ('blastomeres' in self.embryocv_output_tasks) and (embryocv_output['blastomeres'] is not None) and (Fid in embryocv_output['blastomeres'].keys()):
            blastomeres = embryocv_output['blastomeres'][Fid]
            if not skip_process:
                for image_id, cells in blastomeres.items():
                    if len(cells) == 0:
                        continue
                    for cell in cells:
                        segm = np.zeros([500,500], np.uint8)
                        cell['mask'] = cv2.fillPoly(segm, np.array([cell['xy_polygon']], dtype=np.int32), 1)
        else:
            blastomeres = []
            task_not_exist.append('blastomeres')
        # zona segmentation mask in compressed io.BytesIO object format.
        if 'zona' in self.embryocv_output_tasks:
            zona = embryocv_output['zona'][Fid]
            if not skip_process:
                for image_id, val in zona.items():
                    val = Image.open(val)
                    zona[image_id] = np.array(val, dtype='uint8')
        else:
            zona = []

        embryocv_output['pronuclei'] = pronuclei
        embryocv_output['blastomeres'] = blastomeres
        embryocv_output['zona'] = zona
        embryocv_output['missing_tasks'] = task_not_exist

        return embryocv_output

    def preprocess_pronuclei(self, pronuclei_dict, boxes, crop_ROI):
        # input: mask images that matches the number of instances in pronuclei. one mask image per instance.
        pronuclei = {image_file:None for image_file in pronuclei_dict.keys()}
        for frame_idx, (key, value) in enumerate(pronuclei_dict.items()):
            pronuclei_image = np.zeros([500,500])
            if value is not None: # if pronuclei prediction does not exist it will output zero np.array
                for cell in value: # multiple cell instances in one frame
                    pronuclei_image += cell['mask']
            if crop_ROI:
                lower, left, height, width = boxes[frame_idx][1]
                pronuclei_image = pronuclei_image[lower:lower+height, left:left+width]                
            pronuclei[key] = pronuclei_image
        return pronuclei

    def preprocess_blastomeres(self, blastomeres_dict, boxes, crop_ROI):
        # input: mask images that matches number of instances in blastomeres. one mask image per instance.
        blastomeres = {image_file:None for image_file in blastomeres_dict.keys()}
        for frame_idx, (key, value) in enumerate(blastomeres_dict.items()):
            blastomeres_image = np.zeros([500,500])
            if value is not None: # if blastomeres prediction does not exist it will output zero np.array
                for cell in value: # multiple cell instances in one frame
                    blastomeres_image += cell['mask']
            if crop_ROI:
                # crop the blastomere image using the bounding box
                lower, left, height, width = boxes[frame_idx][1] # num_frame x fids (F-,F0,F+) x box [lower, left, height, width]
                blastomeres_image = blastomeres_image[lower:lower+height, left:left+width]  
            blastomeres[key] = blastomeres_image

        return blastomeres

    def parse_embryocv_output(self, embryocv_output, image_files, max_frame=None, crop_ROI=False):
        # this function is used to parse the embryocv_output. It will return None values for int/float types or zeros np.array types for frames that embryocv_output do not exist.
        Fids = sorted(embryocv_output['frag'].keys()) if self.F_per_task['frag'] == 'all' else [self.F_per_task['frag']]
        if 'frag' in self.embryocv_output_tasks:
            frag = []
            for Fid in Fids:
                frag_dict = {image_file:None for image_file in image_files}
                frag_dict = {image_file:embryocv_output['frag'][Fid][image_file] for image_file in embryocv_output['frag'][Fid].keys()}
                frag.append(np.array(list(frag_dict.values())).astype(np.float32))
            frag = np.stack(frag, axis=1)
            frag = np.expand_dims(frag, axis=2)
        else:
            frag=None

        if 'stage_raw' in self.embryocv_output_tasks:
            stage_raw = []
            for Fid in Fids:
                stage_raw_dict = {image_file:None for image_file in image_files}
                stage_raw_dict = {image_file:embryocv_output['stage_raw'][Fid][image_file] for image_file in embryocv_output['stage_raw'][Fid].keys()}
                stage_raw.append(np.array(list(stage_raw_dict.values())).astype(np.float32))
            stage_raw = np.stack(stage_raw, axis=0).transpose(1,0,2)
        else:
            stage_raw = []

        if 'stage_smooth' in self.embryocv_output_tasks:
            stage_smooth = []
            for Fid in Fids:
                stage_smooth_dict = {image_file:None for image_file in image_files}
                stage_smooth_dict = {image_file:embryocv_output['stage_smooth'][Fid][image_file] for image_file in embryocv_output['stage_smooth'][Fid].keys()}
                stage_smooth.append(np.array(list(stage_smooth_dict.values())).astype(np.uint8))
            stage_smooth = np.stack(stage_smooth, axis=1)
            stage_smooth = np.expand_dims(stage_smooth, axis=2)
        else:
            stage_smooth = []

        if 'boxes' in self.embryocv_output_tasks:
            boxes = []
            for Fid in Fids:
                boxes_dict = {image_file:None for image_file in image_files}
                boxes_dict = {image_file:embryocv_output['boxes'][Fid][image_file] for image_file in embryocv_output['boxes'][Fid].keys()}
                boxes.append(np.array(list(boxes_dict.values())))
            boxes = np.stack(boxes, axis=1)
        else:
            boxes = []
        
        video_len = int(image_files[-1].split('.')[0])
        if boxes.shape[0] < video_len:
            print(f'out of range error before preprocess. frame gap: {video_len - boxes.shape[0]}')

        if 'pronuclei' in self.embryocv_output_tasks:
            if 'pronuclei' not in embryocv_output['missing_tasks']:
                pronuclei_dict = {image_file:None for image_file in image_files}
                pronuclei_dict = {image_file:embryocv_output['pronuclei'][image_file] for image_file in embryocv_output['pronuclei'].keys()}
                pronuclei = self.preprocess_pronuclei(pronuclei_dict, boxes, crop_ROI=crop_ROI)
                pronuclei = np.array(list(pronuclei.values())).astype(np.uint8)
            else:
                # return a blank image when the pronuclei prediction does not exist
                pronuclei = np.zeros([len(image_files), 500, 500]).astype(np.uint8)
        else:
            pronuclei = []
        if 'blastomeres' in self.embryocv_output_tasks:
            if 'blastomeres' not in embryocv_output['missing_tasks']:
                blastomeres_dict = {image_file:None for image_file in image_files}
                blastomeres_dict = {image_file:embryocv_output['blastomeres'][image_file] for image_file in embryocv_output['blastomeres'].keys()}
                blastomeres = self.preprocess_blastomeres(blastomeres_dict, boxes, crop_ROI=crop_ROI)
                blastomeres = np.array(list(blastomeres.values())).astype(np.uint8)
            else:
                # return a blank image when blastomeres prediction does not exist
                blastomeres = np.zeros([len(image_files), 500, 500]).astype(np.uint8)
        else:
            blastomeres = []

        if 'zona' in self.embryocv_output_tasks:
            zona_dict = {image_file:None for image_file in image_files}
            zona_dict = {image_file:embryocv_output['zona'][image_file] for image_file in embryocv_output['zona'].keys()}
            zona = np.array(list(zona_dict.values())).astype(np.uint8)
            if self.crop_ROI:
                zona_cropped = np.zeros_like(blastomeres)
                for frame_idx in range(zona_cropped.shape[0]):
                    lower, left, height, width = boxes[frame_idx][1]
                    zona_cropped[frame_idx] = zona[frame_idx][lower:lower+height, left:left+width]
                zona = zona_cropped
        else:
            zona = []
        
        embryocv_parsed = {'frag': frag, 'stage_raw': stage_raw, 'stage_smooth': stage_smooth, 'boxes': boxes, 'pronuclei': pronuclei, 'blastomeres': blastomeres, 'zona': zona}

        if max_frame > 0 and max_frame > len(image_files):
            for key, val in embryocv_parsed.items():
                if key in self.embryocv_output_tasks:
                    embryocv_parsed[key] = pad_np_array(val, max_frame)
        elif max_frame > 0 and max_frame < len(image_files):
            for key, val in embryocv_parsed.items():
                if key in self.embryocv_output_tasks:
                    embryocv_parsed[key] = val[:max_frame]

        return embryocv_parsed

    def get_statistics_of_splits(self):
        embryo_id_sets = {'train':self.embryo_ids_in_train, 'val':self.embryo_ids_in_val, 'test':self.embryo_ids_in_test}

        for split, embryo_ids in embryo_id_sets.items():
            embryo_ids_success = []
            embryo_ids_fail = []
            embryo_ids_not_exists = []
            for embryo_id in embryo_ids:
                slide_id = embryo_id['slide_id']
                well_id = embryo_id['well']
                slide_well_id = os.path.join(slide_id + '_pdb', f"WELL{int(well_id):02}")
                if slide_well_id not in self.EHR_data.keys():
                    embryo_ids_not_exists.append(embryo_id)
                    continue
                EHR_embryo = self.EHR_data[slide_well_id] # all EHR columns are retrieved.
                EHR_outcome = {key:val for key, val in EHR_embryo.items() if key in EHR_field_outcome}
                if EHR_outcome['tot_number_embryos_transferred'] == 0:
                    # Note: this happens because there is data mismatch between annotation.csv and final_concat.csv
                    EHR_outcome['tot_number_embryos_transferred'] = int(embryo_id['transferred'])
                gt_viability = EHR_outcome['children_N'] / EHR_outcome['tot_number_embryos_transferred']
                if gt_viability > 0:
                    embryo_ids_success.append(embryo_id)
                else:
                    embryo_ids_fail.append(embryo_id)
            
            print(f'----- split: {split} -----')
            print(f'total embryos: {len(embryo_ids)}')
            print(f'total embryos with successful pregnancies: {len(embryo_ids_success)}')
            print(f'total embryos with failed pregnancies: {len(embryo_ids_fail)}')
            print(f'total embryos with no EHR data: {len(embryo_ids_not_exists)}')

            treatment_success = set([embryo_id['slide_id'] for embryo_id in embryo_ids_success])
            treatment_fail = set([embryo_id['slide_id'] for embryo_id in embryo_ids_fail])
            print(f'total treatments with successful pregnancies: {len(treatment_success)}')
            print(f'total treatments with failed pregnancies: {len(treatment_fail)}')

        EHR_outcome = {key:val for key, val in EHR_embryo.items() if key in EHR_field_outcome}
        if EHR_outcome['tot_number_embryos_transferred'] == 0:
            # Note: this happens because there is data mismatch between annotation.csv and final_concat.csv
            EHR_outcome['tot_number_embryos_transferred'] = int(embryo_id['transferred'])
        gt_viability = EHR_outcome['children_N'] / EHR_outcome['tot_number_embryos_transferred']

class BatchTransform:
    def __init__(self, 
                 target_size, 
                 mean, 
                 std, 
                 augmentation=True):
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.augmentation = augmentation

    def __call__(self, batch,flip,angle):

        augmented_batch = []
        for img in batch:
            if self.augmentation:
                if flip:
                    img = transforms.functional.hflip(img)
                img = transforms.functional.rotate(img, angle)
            img = transforms.functional.to_tensor(img)
            img = transforms.functional.normalize(img, mean=self.mean, std=self.std)
            augmented_batch.append(img)

        return augmented_batch

class BalancedSampler(Sampler):
    def __init__(self, dataset, sample_length=None):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))

        # Access labels directly from the dataset's 'embryo_labels' attribute
        self.label_to_indices = {0: [], 1: []}
        for i in self.indices:
            label =  dataset.embryo_labels[i]
            self.label_to_indices[label].append(i)
        
        # Shuffle the indices in each label group
        random.shuffle(self.label_to_indices[0])
        random.shuffle(self.label_to_indices[1])

        if sample_length == None:
            sample_length = max(len(self.label_to_indices[0]), len(self.label_to_indices[1]))

        self.balanced_indices = []
        for i in range(sample_length):
            index0 = i % len(self.label_to_indices[0])
            index1 = i % len(self.label_to_indices[1])

            if index0 == 0:
                random.shuffle(self.label_to_indices[0])
            if index1 == 0:
                random.shuffle(self.label_to_indices[1])

            self.balanced_indices.append(self.label_to_indices[0][index0])
            self.balanced_indices.append(self.label_to_indices[1][index1])