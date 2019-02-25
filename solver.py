import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from model import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss
import pdb
import json


class Solver():
	def __init__(self, config):
		self.data_dir = config.data_dir
		self.log = config.do_log
		self.split=config.split
		
		self.epochs = config.epochs
		self.batch_size=config.batch_size
		self.lr = config.lr
		self.momentum = config.momentum
		self.weight_decay = config.weight_decay
		
		with open('config.json', 'r') as f:
			config = json.load(f)
		
		self.boundary = config["boundary"]
		self.class_list = config["class_list"]
		self.anchors = config["anchors"]
		self.num_anchors = len(self.anchors)//2
		self.num_classes = len(self.class_list) + 1

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.buildModel()

	def buildModel(self):
		self.model = ComplexYOLO()
		self.model.to(self.device)
	
	def buildDataset(self, toggle):
		self.dataset = KittiDataset(root_dir=self.data_dir,
									class_list=self.class_list,
									boundary=self.boundary,
									split=self.split)
		
		if toggle=='val':
			self.dataset.toggleVal()

	def buildLogger(self):
		pass

	def train(self):
		self.buildDataset('train')

		opt = optim.SGD(self.model.parameters(), 
						lr =self.lr, 
						momentum = self.momentum, 
						weight_decay = self.weight_decay)
		region_loss = RegionLoss(num_classes = self.num_classes, num_anchors=self.num_anchors)
		region_loss = region_loss.to(self.device)
		# Dataloader
		data_loader = DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=False)
		
		for epoch in range(self.epochs):
			
			for idx, data in enumerate(data_loader):
				pdb.set_trace()
				rgb_map = data[0]
				target = data[1]
				rgb_map = rgb_map.float().to(self.device)

				output = self.model(rgb_map)
				loss = region_loss(output, target)
				loss.backward()
				opt.step()		
		

	def test(self):
		pass

	

