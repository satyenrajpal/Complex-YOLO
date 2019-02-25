"""
    Defines class to load the KITTI dataset. 
"""
from __future__ import division
import os
import os.path
import torch
import numpy as np
import cv2
import math
import json 

from utils import *

# Load json configs 


def removePoints(point_cloud, boundary_cond):
    
    # Boundary condition
    minX = boundary_cond['minX'] ; maxX = boundary_cond['maxX']
    minY = boundary_cond['minY'] ; maxY = boundary_cond['maxY']
    minZ = boundary_cond['minZ'] ; maxZ = boundary_cond['maxZ']
    
    # Remove the point out of range x,y,z
    mask = np.where(
            (point_cloud[:, 0] >= minX) & 
            (point_cloud[:, 0] <= maxX) & 
            (point_cloud[:, 1] >= minY) & 
            (point_cloud[:, 1] <= maxY) & 
            (point_cloud[:, 2] >= minZ) & 
            (point_cloud[:, 2] <= maxZ)
            )
    point_cloud = point_cloud[mask]
    return point_cloud

"""
TODO:
 - [x] - Generate train set and test set in here 
 - [x] - Remove config.json dependency in utils
"""

class KittiDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, class_list, boundary, set = 'lidar', split = 0.9):
        
        self.root_dir = root_dir
        self.data_path = os.path.join(self.root_dir, 'training')
        self.lidar_path = os.path.join(self.data_path, "velodyne/")
        self.image_path = os.path.join(self.data_path, "image_2/")
        self.calib_path = os.path.join(self.data_path, "calib/")
        self.label_path = os.path.join(self.data_path, "label_2/")
        # with open(os.path.join(self.data_path, '%s.txt' % set)) as f:
        #     self.file_list = f.read().splitlines()
        self.toggle = 'train'
        self.class_list = class_list
        self.boundary = boundary
        self.preprocess(split)

    def preprocess(self, split):
        # Generate train set and validation set here
        file_list = [f for f in os.listdir(self.calib_path) if os.path.isfile(os.path.join(self.calib_path, f))]
        self.train_length = int(len(file_list) *split )
        self.val_length = len(file_list) - self.train_length
        print("Training on {} samples. Validating on {} samples".format(self.train_length, self.val_length))
    
    def toggleVal(self):
        # Implemented this to convert this class into test as well
        self.toggle = 'val'

    def __getitem__(self, i):
        
        file_num = i if self.toggle=='train' else self.train_length+i
        file_num = str(file_num).zfill(6)

        lidar_file = os.path.join(self.lidar_path, file_num+'.bin')
        calib_file = os.path.join(self.calib_path, file_num+'.txt')
        label_file = os.path.join(self.label_path, file_num+'.txt')
        image_file = os.path.join(self.image_path, file_num+'.png')
        
        if self.toggle == 'train' or self.toggle=='val':
            calib = load_kitti_calib(calib_file)  
            target = get_target(label_file, calib['Tr_velo2cam'], self.boundary, self.class_list)
        
            # load point cloud data (x, y, z, r) r-> reflectance value
            point_cloud = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
            
            pruned_points = removePoints(point_cloud, self.boundary)
            data = makeBVFeature(pruned_points, self.boundary, 40 / 512)
            data = np.transpose(data, (2,0,1))
            return data , target

        elif self.toggle == 'test':
            NotImplemented
        
        else:
            raise ValueError('the type invalid')

    def __len__(self):
        return self.train_length if 'train' in self.toggle else self.val_length
