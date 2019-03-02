"""
    Util scripts for building features, fetching ground truths, computing IoU, etc. 
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import cv2
import math
import json
import pdb

# Load configs fron json
with open('config.json', 'r') as f:
    config = json.load(f)

# Classes of objects and anchors
class_list = config["class_list"]
anchors = config["anchors"]

def makeBVFeature(PointCloud_, BoundaryCond, Discretization):
    
    # 1024 x 1024 x 3
    Height = 1024 + 1
    Width = 1024 + 1

    # Discretize Feature Map
    PointCloud = np.copy(PointCloud_)
    PointCloud[:,0] = np.int_(np.floor(PointCloud[:,0] / Discretization)) # <- X
    PointCloud[:,1] = np.int_(np.floor(PointCloud[:,1] / Discretization) + Width / 2) # <- Y ranges between [-range, range]    

    # sort 3times with respect to z, y and x respectively
    indices = np.lexsort((-PointCloud[:,2], PointCloud[:,1], PointCloud[:,0]))
    PointCloud = PointCloud[indices]

    # Height Map
    heightMap = np.zeros((Height, Width))

    # Remove duplicate points
    _, indices, counts = np.unique(PointCloud[:, 0:2], axis = 0, return_index = True, return_counts=True)
    PointCloud_uniq = PointCloud[indices]

    # Counts-> Number of duplicate elements for each index in the unique array
    # X in the data is front -> image coordinates y
    # Y in the data is left -> image coordinates x
    heightMap[np.int_(PointCloud_uniq[:, 0]), np.int_(PointCloud_uniq[:, 1])] = PointCloud_uniq[:, 2]

    # Intensity Map & DensityMap ##########################
    
    # _, indices, counts = np.unique(PointCloud[:, 0:2], axis = 0, return_index = True, return_counts = True)
    # PointCloud_top = PointCloud[indices]
    # Changed PointCloud_top -> PointCLoud_uniq
    
    # Intensity Map
    intensityMap = np.zeros((Height, Width))
    intensityMap[np.int_(PointCloud_uniq[:, 0]), np.int_(PointCloud_uniq[:, 1])] = PointCloud_uniq[:, 3]

    # Density Map
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    densityMap = np.zeros((Height, Width))
    densityMap[np.int_(PointCloud_uniq[:, 0]), np.int_(PointCloud_uniq[:, 1])] = normalizedCounts
    

    # RGB channels respectively
    RGB_Map = np.zeros((Height,Width, 3))
    RGB_Map[:,:,0] = densityMap
    RGB_Map[:,:,1] = heightMap
    RGB_Map[:,:,2] = intensityMap
    
    # save = np.zeros((512, 1024, 3))
    save = RGB_Map[0:512, 0:1024, :]
    return save


def get_target(label_file, Tr, boundary, class_list):
    """ Make target vector (class, x, y, w, l, im, re) """
    target = np.zeros([50, 7], dtype = np.float32)
    minX = boundary['minX'] ; maxX = boundary['maxX']
    minY = boundary['minY'] ; maxY = boundary['maxY']
    minZ = boundary['minZ'] ; maxZ = boundary['maxZ']
    
    with open(label_file, 'r') as f:
        lines = f.readlines() 

    num_obj = len(lines)
    index = 0
    for j in range(num_obj):
        obj = lines[j].strip().split(' ')
        obj_class = obj[0].strip()
        if obj_class in class_list:

            # Get target 3D object location x, y
            t_lidar , box3d_corner = box3d_cam_to_velo(obj[8:], Tr)
            location_x = t_lidar[0][0]          
            location_y = t_lidar[0][1]            
            
            if (location_x > minX) and (location_x < maxX) and (location_y > minY) and (location_y < maxY):
                
                # Make sure target inside the covering area (0,1)
                # x and y interchange?
                target[index][2] = location_x / 40 # X is along height
                
                # Should put this in [0,1] ,so divide max_size 80 m
                target[index][1] = (location_y + 40)/80 # Y is along width
                obj_width = obj[9].strip()
                obj_length = obj[10].strip()
                target[index][3] = float(obj_width) / 80
                target[index][4] = float(obj_length) / 40

                # Get target Observation angle of object, ranging [-pi .. pi]
                obj_alpha = obj[3].strip()
                assert target[index][1] <= 1 
                assert target[index][2] <= 1
                # Im axis
                target[index][5] = math.sin(float(obj_alpha))

                # Re axis
                target[index][6] = math.cos(float(obj_alpha))
                for i in range(len(class_list)):
                    if obj_class == class_list[i]:
                        target[index][0] = i
                index = index + 1
    
    return target


def box3d_cam_to_velo(box3d, Tr):

    def project_cam2velo(cam, Tr):
        T = np.zeros([4, 4], dtype = np.float32)
        T[:3, :] = Tr
        T[3, 3] = 1
        T_inv = np.linalg.inv(T)
        lidar_loc_ = np.dot(T_inv, cam)
        lidar_loc = lidar_loc_[:3]
        return lidar_loc.reshape(1, 3)

    def ry_to_rz(ry):
        angle = -ry - np.pi / 2
        if angle >= np.pi:
            angle -= np.pi
        if angle < -np.pi:
            angle = 2 * np.pi + angle
        return angle

    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]
    cam = np.ones([4, 1])
    cam[0] = tx
    cam[1] = ty
    cam[2] = tz
    t_lidar = project_cam2velo(cam, Tr)
    Box = np.array([[-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
                    [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
                    [0, 0, 0, 0, h, h, h, h]])
    rz = ry_to_rz(ry)
    rotMat = np.array([
        [np.cos(rz), -np.sin(rz), 0.0],
        [np.sin(rz), np.cos(rz), 0.0],
        [0.0, 0.0, 1.0]])
    velo_box = np.dot(rotMat, Box)
    cornerPosInVelo = velo_box + np.tile(t_lidar, (8, 1)).T
    box3d_corner = cornerPosInVelo.transpose()
    return t_lidar , box3d_corner.astype(np.float32)


def load_kitti_calib(calib_file):
    """
    load projection matrix
    """
    with open(calib_file) as fi:
        lines = fi.readlines()
        assert (len(lines) == 8)
    obj = lines[0].strip().split(' ')[1:]
    P0 = np.array(obj, dtype=np.float32)
    obj = lines[1].strip().split(' ')[1:]
    P1 = np.array(obj, dtype=np.float32)
    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
    obj = lines[6].strip().split(' ')[1:]
    Tr_imu_to_velo = np.array(obj, dtype=np.float32)
    return {'P2': P2.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def bbox_iou(box1, box2, x1y1x2y2 = True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0] - box1[2] / 2.0, box2[0] - box2[2] / 2.0)
        Mx = max(box1[0] + box1[2] / 2.0, box2[0] + box2[2] / 2.0)
        my = min(box1[1] - box1[3] / 2.0, box2[1] - box2[3] / 2.0)
        My = max(box1[1] + box1[3] / 2.0, box2[1] + box2[3] / 2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea


def bbox_ious(boxes1, boxes2, x1y1x2y2 = True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes
    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1 - boxes[i][4]                
    _, sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2 = False) > nms_thresh:
                    box_j[4] = 0
    return out_boxes


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)
