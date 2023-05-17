import os
import os.path as op
import sys
import cv2
import random
from PIL import Image, ImageDraw
import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np
import json
from .utils import generate_gt,load_ann_tt,load_ann
from .aug_e2e import PSSAugmentation_e2e
from scipy import io as sio
import imageio
import math
import re

from utils.str_label_converter import StrLabelConverter


class MultiStageTextLoader(data.Dataset):
    def __init__(self, config, converter, is_training):
        self.root = config.TEST.ROOT
        self.config = config
        self.is_training = is_training
        self.image_paths = []
        self.gt_paths = []

        self.split = "Train"
        if self.is_training:
            self.augmentation = PSSAugmentation_e2e(config.TEST.MIN_SIZE)
        else:
            self.split = "Test"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.converter = converter
        if self.config.TEST.NAME == "tt":
            self.get_all_samples_tt()
        elif self.config.TEST.NAME == "ic15":
            self.get_all_samples_ic15()
        elif self.config.TEST.NAME == "ctw":
            self.get_all_samples_ctw()

    def get_all_samples_tt(self): 

        img_dir = op.join(self.root, "Images", self.split)
        for sample in os.listdir(img_dir):
            if sample.strip().split('.')[-1] == 'jpg':
                self.image_paths.append(os.path.join(self.root, "Images", self.split, sample))
                if self.is_training:
                    self.gt_paths.append(os.path.join(self.root, 'gts', 'Train', 'poly_gt_' + sample.strip().split('.')[0] + '.txt'))

        self.targets = load_ann_tt(self.gt_paths, self.config.MAX_LENGTH)
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def get_all_samples_ic15(self):
        img_dir = op.join(self.root + 'test_images')
        for sample in os.listdir(img_dir):
            if sample.strip().split('.')[-1] == 'jpg':
                self.image_paths.append(os.path.join(img_dir, sample))
                if self.is_training:
                    self.gt_paths.append(os.path.join(self.root, "test" + '_gts', sample.strip() + '.txt'))
        self.targets = load_ann(self.gt_paths,self.config.MAX_LENGTH)
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)
    
    def get_all_samples_ctw(self):
        ## check datasets#/data/wujingjing/data/ctw1500/train/text_image/
        img_dir = op.join(self.root, "ctw1500", "test" , 'text_image')
        print(img_dir)
        for sample in os.listdir(img_dir):
            if sample.strip().split('.')[-1] == 'jpg':
                self.image_paths.append(os.path.join(img_dir, sample))
                if self.is_training:
                    self.gt_paths.append(os.path.join(self.root, "ctw1500", 'gts_train', sample.strip() + '.txt'))

    def __len__(self):
        return len(self.image_paths)
    
    def resize_fix_max(self,img, image_path):
        ori_height, ori_width, = img.shape[0], img.shape[1]
        if ori_height < ori_width:
            test_width = self.config.TEST.MAX_SIZE
            test_height = int(test_width*ori_height*1.0/ori_width)
            pad_width = self.config.TEST.MAX_SIZE
            pad_height = int(math.ceil(test_width * ori_height * 1.0 / ori_width / 128) * 128)
        else:
            test_height = self.config.TEST.MAX_SIZE
            test_width = int(test_height*ori_width*1.0/ori_height)
            pad_height = self.config.TEST.MAX_SIZE
            pad_width = int(math.ceil(test_height * ori_width * 1.0 / ori_height / 128) * 128)
        
        img_resized = cv2.resize(img.copy(), (test_width, test_height))
        img_padded = np.zeros((pad_height, pad_width, 3))

        img_padded[:test_height, :test_width, :] = img_resized
        img_padded = img_padded.astype(np.uint8)
        img_padded = self.transformer(img_padded)
        return img_padded, image_path, ori_height, ori_width, test_height, test_width
    
    def resize_fix_min(self, img, image_path):
        ori_height, ori_width, = img.shape[0], img.shape[1]

        if ori_height < ori_width:
            test_height = self.config.TEST.MIN_SIZE
            test_width = int(test_height*ori_width*1.0/ori_height)
            pad_height = self.config.TEST.MIN_SIZE
            pad_width = int(math.ceil(test_height * ori_width * 1.0 / ori_height / 128) * 128)
        else:
            test_width = self.config.TEST.MIN_SIZE
            test_height = int(test_width * ori_height * 1.0 / ori_width)
            pad_width = self.config.TEST.MIN_SIZE
            pad_height = int(math.ceil(test_width * ori_height * 1.0 / ori_width / 128) * 128)
        img_resized = cv2.resize(img.copy(), (test_width, test_height))
        img_padded = np.zeros((pad_height, pad_width, 3))

        img_padded[:test_height, :test_width, :] = img_resized
        img_padded = img_padded.astype(np.uint8)
        img_padded = self.transformer(img_padded)
        return img_padded, image_path, ori_height, ori_width, test_height, test_width
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = imageio.imread(image_path, pilmode='RGB')

        if self.is_training:
            target = self.targets[index]
            img, boxes, tags, text = self.augmentation(img, target['polys'], target['tags'], target['label'])
            aug_img, keep_polys, ins_masks, num_of_polys, training_mask, score_map, loc_map, sampled_map, text_labels, text_lengths, center_points, num_points = generate_gt(
                img, boxes, tags, text, self.converter, self.args)
            aug_img = aug_img.astype(np.uint8)
            aug_img = self.transformer(aug_img)
            return aug_img, keep_polys, ins_masks, num_of_polys, training_mask, score_map, loc_map, sampled_map, text_labels, text_lengths, center_points, num_points, image_path
        else:
            if self.config.TEST.FIX_MAX == False:
                return self.resize_fix_min(img, image_path)
            else:
                return self.resize_fix_max(img, image_path)