"""
    Defines the objective function (loss function) for the model. 
"""

import time
import torch
import logging
import itertools
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils import *


def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh):
    
    # nTrueBox = 50, nA = 5, nC = 8
    nB = target.size(0)
    nTrueBox = target.data.size(1)
    nA = num_anchors
    nC = num_classes
    anchor_step = len(anchors) / num_anchors
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coord_mask = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW) 
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    tl         = torch.zeros(nB, nA, nH, nW)
    tim        = torch.zeros(nB, nA, nH, nW)
    tre        = torch.zeros(nB, nA, nH, nW)
    tconf      = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 

    nAnchors = nA * nH * nW
    nPixels  = nH * nW
    loss_iou = 0.0

    # Target: nB, 50, 7 -> class, x, y, w, l, im, re 
    # Pred boxes.size =  nB*nA*nH*nW x 6
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b * nAnchors:(b + 1) * nAnchors].t() # (nAnchors, 6)^T
        cur_ious = torch.zeros(nAnchors)
        
        for t in range(nTrueBox):
            if target[b][t][1] == 0: # If x==0 # If there are no more objects
                break

            # nW = 32, nH = 16
            gx = target[b][t][1] * nW
            gy = target[b][t][2] * nH
            gw = target[b][t][3] * nW
            gl = target[b][t][4] * nH
            gim = target[b][t][5]
            gre = target[b][t][6]
            cur_gt_boxes = torch.FloatTensor([gx, gy, gw, gl]).repeat(nAnchors, 1).t()
            # Compute the max for all possible anchors. cur_ious.size() -> nAnchors
            cur_ious = torch.max(cur_ious, bbox_ious(cur_pred_boxes, cur_gt_boxes, x1y1x2y2 = False))
        
        # Doubtful about this
        # likely_boxes = torch.masked_select(cur_pred_boxes, (cur_ious > sil_thresh).byte().repeat(6, 1))
        
        # if likely_boxes.size() != torch.Size([0]):
        #     likely_boxes = likely_boxes.view(6, -1)
        #     all_combo = itertools.product(likely_boxes.transpose(0, 1), likely_boxes.transpose(0, 1))
        #     for combo in all_combo:
        #         loss_iou += bbox_iou(combo[0], combo[1], x1y1x2y2 = False)
        #     loss_iou -= float(likely_boxes.shape[1])

        conf_mask = conf_mask.view(nB, nAnchors)
        conf_mask[b][cur_ious > sil_thresh] = 0 # All nAnchors <sil_thresh=no_object_scale

    nGT = 0
    nCorrect = 0

    for b in range(nB):
        for t in range(nTrueBox):
            if target[b][t][1] == 0:
                break
            
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1
            min_dist = 10000

            gx = target[b][t][1] * nW
            gy = target[b][t][2] * nH
            gi = int(gx)
            gj = int(gy)
            gw = target[b][t][3] * nW
            gl = target[b][t][4] * nH
            gim = target[b][t][5]
            gre = target[b][t][6]
            gt_box = [0, 0, gw, gl]
            
            # Find best bounding box -> highest IOU
            for n in range(nA):
                aw = anchors[int(anchor_step * n)]
                ah = anchors[int(anchor_step * n + 1)]
                anchor_box = [0, 0, aw, ah]
                iou  = bbox_iou(anchor_box, gt_box, x1y1x2y2 = False)

                if anchor_step == 4:
                    ax = anchors[anchor_step * n + 2]
                    ay = anchors[anchor_step * n + 3]
                    dist = pow(((gi + ax) - gx), 2) + pow(((gj + ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            index = b * nAnchors + best_n * nPixels + gj * nW + gi
            pred_box = [pred_boxes[index][0], pred_boxes[index][1], pred_boxes[index][2], pred_boxes[index][3]]
            # pdb.set_trace()
            
            coord_mask[b][best_n][gj][gi] = 1 
            cls_mask[b][best_n][gj][gi] = 1
            conf_mask = conf_mask.view(nB, nA, nH, nW)
            conf_mask[b][best_n][gj][gi] = object_scale

            tx[b][best_n][gj][gi] = target[b][t][1] * nW - gi
            ty[b][best_n][gj][gi] = target[b][t][2] * nH - gj
            tw[b][best_n][gj][gi] = np.log(gw / anchors[int(anchor_step * best_n)])
            tl[b][best_n][gj][gi] = np.log(gl / anchors[int(anchor_step * best_n + 1)])
            tim[b][best_n][gj][gi] = target[b][t][5]
            tre[b][best_n][gj][gi] = target[b][t][6]
            
            gt_box = [gx, gy, gw, gl]
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2 = False)
            tconf[b][best_n][gj][gi] = iou
            tcls[b][best_n][gj][gi] = target[b][t][0]
            
            if iou > 0.5:
                nCorrect = nCorrect + 1
    
    conf_mask = conf_mask.view(nB, nA, nH, nW)
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, tl, tim, tre, tconf, tcls


class RegionLoss(nn.Module):
    def __init__(self, num_classes = 8, num_anchors = 5):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = int(len(anchors) / num_anchors)
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 10
        self.class_scale = 1
        self.thresh = 0.6


    def forward(self, output, target):

        # TODO: Change to '.device'
        # output: B x As * (6 + 1 + num_classes) * H * W
        # import pdb;pdb.set_trace()

        # Create separate function for predicted boxes
        ###########################################################
        t0 = time.time()
        
        nB = output.data.size(0)  # Batch Size
        nA = self.num_anchors     # num_anchors = 5
        nC = self.num_classes     # num_classes = 8
        nH = output.data.size(2)  # nH  16
        nW = output.data.size(3)  # nW  32

        output = output.view(nB, nA, (6 + 1 + nC), nH, nW)
        
        # Get X, Y (center), Width, height (of the Bounding Box), Im, re axis (orientation), and confidence (Pr(object))
        x = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
        y = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
        w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
        l = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
        im = output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW)
        re = output.index_select(2, Variable(torch.cuda.LongTensor([5]))).view(nB, nA, nH, nW)
        conf = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([6]))).view(nB, nA, nH, nW))
        
        # Obtain prob of classes (7 -> 7 + nC)
        cls_prob = output.index_select(2, Variable(torch.linspace(7, 7 + nC - 1, nC).long().cuda()))
        cls_prob = cls_prob.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)
        
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(6, nB * nA * nH * nW)
        
        # Grid positions for X and Y (c_x and c_y in the paper)
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
        
        # Get prior anchor boxes width and length
        anchor_w = torch.Tensor(anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([0])).cuda()
        anchor_l = torch.Tensor(anchors).view(nA, self.anchor_step).index_select(1, torch.LongTensor([1])).cuda()
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        anchor_l = anchor_l.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
        
        # predicted boxes are (c_x + x, c_y + y, w * anchor prior, l * anchor_prior, im, re) 
        pred_boxes[0] = x.data.view(nB * nA * nH * nW).cuda() + grid_x
        pred_boxes[1] = y.data.view(nB * nA * nH * nW).cuda() + grid_y
        pred_boxes[2] = torch.exp(w.data).view(nB * nA * nH * nW).cuda() * anchor_w
        pred_boxes[3] = torch.exp(l.data).view(nB * nA * nH * nW).cuda() * anchor_l
        pred_boxes[4] = im.data.view(nB * nA * nH * nW).cuda()
        pred_boxes[5] = re.data.view(nB * nA * nH * nW).cuda()
        pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, 6)) # Why?
        ###########################################################
        t2 = time.time()

        # Problem here
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, tl, tim, tre, tconf,tcls = \
        build_targets(pred_boxes, target.data, self.anchors, nA, nC, \
        nH, nW, self.noobject_scale, self.object_scale, self.thresh)
        
        cls_mask = (cls_mask == 1)
        nProposals = int(torch.sum(torch.gt(conf, 0.25)))

        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        tl    = Variable(tl.cuda())
        tim   = Variable(tim.cuda())
        tre   = Variable(tre.cuda())
        tconf = Variable(tconf.cuda())
        cls_mask = cls_mask.view(nB * nA * nH * nW)
        tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda()) # Selects only those indices where cls_mask==1

        coord_mask = Variable(coord_mask.cuda())
        conf_mask  = Variable(conf_mask.cuda())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1, nC).cuda())
        cls_prob   = cls_prob[cls_mask].view(-1, nC)
        t3 = time.time()

        loss_x = self.coord_scale * nn.MSELoss(reduction = 'sum')(x * coord_mask, tx * coord_mask)
        loss_y = self.coord_scale * nn.MSELoss(reduction = 'sum')(y * coord_mask, ty * coord_mask)
        loss_w = self.coord_scale * nn.MSELoss(reduction = 'sum')(w * coord_mask, tw * coord_mask)
        loss_l = self.coord_scale * nn.MSELoss(reduction = 'sum')(l * coord_mask, tl * coord_mask)
        loss_conf = nn.MSELoss(reduction = 'sum')(conf * conf_mask, tconf * conf_mask)
        loss_cls = self.class_scale * nn.CrossEntropyLoss(reduction = 'sum')(cls_prob, tcls)
        
        loss_im = self.coord_scale * nn.MSELoss(reduction = 'sum')(im * coord_mask, tim * coord_mask)
        loss_re = self.coord_scale * nn.MSELoss(reduction = 'sum')(re * coord_mask, tre * coord_mask)
        loss_Euler = loss_im + loss_re
        
        loss = loss_x + loss_y + loss_w + loss_l + loss_conf + loss_cls + loss_Euler
        # loss_history[epoch, batch_idx, :] = [loss_x, loss_y, loss_w, loss_l, loss_conf, loss_cls, loss_Euler, loss_iou]
        
        # t4 = time.time()
        if False:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        # logging.info('nGT %d, recall %d, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, Euler %f, total %f' % (nGT, nCorrect, nProposals, loss_x.data, loss_y.data, loss_w.data, loss_l.data, loss_conf.data, loss_cls.data, loss_Euler.data, loss.data))
        return loss, nCorrect, nGT
