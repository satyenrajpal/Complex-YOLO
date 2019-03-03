import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from model import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss
import pdb
import json


class Solver():
	def __init__(self, args):
		self.data_dir = args.data_dir
		self.log = args.do_log
		self.split=args.split
		
		self.epochs = args.epochs
		self.batch_size=args.batch_size
		self.lr = args.lr
		self.momentum = args.momentum
		self.weight_decay = args.weight_decay
		
		with open('config.json', 'r') as f:
			config = json.load(f)
		
		self.boundary = config["boundary"]
		self.class_list = config["class_list"]
		self.anchors = config["anchors"]
		self.num_anchors = len(self.anchors)//2
		self.num_classes = len(self.class_list) + 1
		self.save_dir = args.save_dir
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.buildModel()

	def buildModel(self):
		self.model = ComplexYOLO()
		self.model.to(self.device)
	
	def buildDataset(self, phase):
		self.dataset = KittiDataset(root_dir=self.data_dir,
									class_list=self.class_list,
									boundary=self.boundary,
									split=self.split)
		
		if phase=='val':
			self.dataset.toggleVal()

	def buildLogger(self):
		pass

	def train(self):
		self.buildDataset('train')

		opt = optim.SGD(self.model.parameters(), 
						lr =self.lr)

		region_loss = RegionLoss(num_classes = self.num_classes, num_anchors=self.num_anchors)
		region_loss = region_loss.to(self.device)
		# Dataloader
		data_loader = DataLoader(self.dataset, self.batch_size, shuffle=True, pin_memory=False)
		log_step = 10
		best_loss = 10000
		self.model.train()	
		for epoch in range(self.epochs):
			running_loss = 0
			running_nCorrect = 0
			running_nGT = 0

			for idx, data in enumerate(data_loader):
				# pdb.set_trace()
				rgb_map = data[0]
				target = data[1]
				
				rgb_map = rgb_map.float().to(self.device)

				output = self.model(rgb_map)
				loss, nCorrect, nGT = region_loss(output, target)
				loss.backward()
				running_loss+=loss.item()
				running_nCorrect+=nGT
				running_nCorrect+=nCorrect

				if idx % log_step==0:
					mean_loss = running_loss/log_step
					print("Epoch: {}, Loss: {m:=5.4f}".format(epoch, m=mean_loss))
					print("nCorrect = {m:=5.4f}, nGT = {p:=5.4f}".format(m=running_nCorrect/log_step, 
																		 p=running_nGT/log_step))
					if mean_loss<best_loss:
						best_loss = mean_loss
						path = os.path.join(self.save_dir, 'ep-{}-{m:=5.4f}.pth'.format(epoch, m=mean_loss))
						torch.save(self.model.state_dict(), path)
						print("Saved model for {} epoch\n".format(epoch))
					
					running_loss = 0
					running_nCorrect = 0
					running_nGT = 0
				opt.step()		
		

	def test(self):
		pass

	

