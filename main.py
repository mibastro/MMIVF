import os
import csv
import numpy as np
import pickle
import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from models.vivit_cv import ViViT_CV
from dataloader.multimodal_loader import MultiModalLoader, BalancedSampler
import pytorch_lightning as pl
import random


if __name__ == "__main__":
    # dataloader settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/jskim/Datasets/IVF', help='path to the video dataset')
    parser.add_argument('--video_dir', default='cropped_videos_transferred_w_results', help='video foldername')
    parser.add_argument('--embryocv_dir', default='videos_transferred_output_from_embryovision_per_well', help='embryocv foldername')
    parser.add_argument('--root_EHR', default='/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/jskim/Datasets/IVF/EHR/EHR_Yong', help='path to the EHR dataset')
    parser.add_argument('--matching_filename', default='EHR/EHR_Helen/Annotation.csv', help='csv file to link the EHR data and videos')
    parser.add_argument('--EHR_filename', default='final_concatenated_timings_and_EHR.csv', help='EHR filename')

    # model inputs
    parser.add_argument('--load_video', type=bool, default=True,help="load time-lapse videos")
    parser.add_argument('--load_EHR', type=bool, default=True,help="load EHR") 
    parser.add_argument('--load_EHRcv', type=bool, default=True,help="load interpretable features (embryocv -> tabular data)")
    parser.add_argument('--load_embryocv', type=bool, default=True,help="load embryocv")
    parser.add_argument('--crop_ROI',type=bool, default=True, help='crop zona region')
    parser.add_argument('--frame_size', type=int,default=90, help='number of frame inputs to the model')
    parser.add_argument('--fix_step', type=int, default=4, help='frame interval for frame sampling')
    parser.add_argument('--frame_info_dim', type=int,default=14, help='input dimension for frame information (non-image embryoCV features, e.g., fragmentation prediction and stage prediction.)')
    
    # model settings
    parser.add_argument('--dim', type=int,default=192, help='hidden dimension')
    parser.add_argument('--spatial_depth', type=int,default=6, help='spatial_depth')
    parser.add_argument('--temporal_depth', type=int,default=4, help='temporal_depth')
    parser.add_argument('--depth', type=int,default=4, help='depth for vit')
    parser.add_argument('--heads', type=int,default=8, help='heads')
    parser.add_argument('--Bap_dim', type=int,default=256, help='mlp_dim')

    # training settings
    parser.add_argument('--pre_train', type=str, default="", help='pre-trained model path')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--epochs', type=int, default=10, help='max epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--augment', type=bool, default=True, help='Flip and rotation augmentation')
    parser.add_argument('--weight_decay', type=float,default=0.0, help='mlp_dim')

    # train/test settings
    parser.add_argument('--mode', type=str, default="train", help='train or test')
    
    # evaluation settings
    parser.add_argument('--f1_threshold', type=float, default=0.15, help='threshold for embryo f1 score calculation')

    args = parser.parse_args()

    args.channel = 0
    if args.load_embryocv:
        args.channel += 3
    if args.load_video:
        args.channel += 1

    print(args)

    args.name = "video=" + str(args.load_video) + '_' + \
                "embryocv=" + str(args.load_embryocv) + '_' + \
                "EHR=" + str(args.load_EHR) + '_' +  \
                "EHRcv=" + str(args.load_EHRcv)
    print(args.name)

    args.global_info_dim = 0
    if args.load_EHR: # EHR data features
        args.global_info_dim += 8
    if args.load_EHRcv: # BlastAssist features
        args.global_info_dim += 39

    model = ViViT_CV(args)
    if args.pre_train != "":
        model = ViViT_CV.load_from_checkpoint(checkpoint_path=args.pre_train, args=args)
    print(model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val_auc",
        mode="max",
        save_last=True,
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    if args.mode == "train":
        logger = pl.loggers.TensorBoardLogger("log", args.name)
    else:
        logger = pl.loggers.TensorBoardLogger("log", args.name + "_test")
    trainer = pl.Trainer(
        accumulate_grad_batches=4,
        accelerator="gpu",
        devices=args.gpus,
        logger=logger,
        val_check_interval=0.5,
        max_epochs=args.epochs,
        callbacks=callbacks,
        num_sanity_val_steps=0
        )
    
    if args.mode == "train":
        print(" ")
        print("train_dataset info")
        train_dataset = MultiModalLoader(args,split='train', augmentation=args.augment, train=True)
        print(" ")
        print("val_dataset info")
        val_dataset = MultiModalLoader(args, split='val', augmentation=False, train=False,)
        train_sampler = BalancedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        test_dataset = MultiModalLoader(args,split='test', augmentation=False, train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        trainer.test(model, test_dataloader)