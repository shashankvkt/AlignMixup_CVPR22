import os 
import sys
import numpy as np
import random
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F
from torchvision import datasets,transforms
from torchvision.utils import save_image
from PIL import Image
import glob
import numpy 

textFile = 'utils/wnids.txt'
classes_idx = []
classes = {}
cnt = 0
with open(textFile) as f:
	for line in f:
		classes_idx.append(line.rstrip('\n'))
		classes[line.rstrip('\n')] = cnt
		cnt += 1



class dataloader_val(Dataset):
	def __init__(self, ImagePth, valtxtfile,transform=None):
		self.ImagePth = ImagePth
		self.valtxtfile = valtxtfile
		self.transform = transform
		
		imagelist = []
		labelList = []
		imgname = []
		classId = []
		
		
		with open(self.valtxtfile) as f:
			for line in f:
				imagelist.append(self.ImagePth + '/' + line.split()[0])
				labelList.append(classes[line.split()[1]])
				
		self.files = imagelist
		self.labels = labelList		
			  
			
	def __getitem__(self,index):
		image = Image.open(self.files[index])
		img_tmp = (numpy.array(image))
		if(img_tmp.ndim == 2):
			img_tmp = img_tmp.reshape((img_tmp.shape[0], img_tmp.shape[1], 1))
			img_tmp = numpy.concatenate([img_tmp, img_tmp, img_tmp], axis=2)

		image = Image.fromarray(img_tmp)
		if(self.transform):
			image = self.transform(image)
		
		label = self.labels[index]

		
		return image, label

	def __len__(self):
		return len(self.files)


class dataloader_train(Dataset):
	def __init__(self, ImagePth, transform=None):
		self.ImagePth = ImagePth
		self.transform = transform
		
		imagelist = []
		labelList = []
		
		classes_idx = sorted(os.listdir(self.ImagePth))
		
		for i in range(len(classes)):
			for filename in glob.glob(self.ImagePth + '/' + classes_idx[i] + '/images/' +'*.JPEG'):

				imagelist.append(filename)
				labelList.append(classes[classes_idx[i]])
				
		
		self.files = imagelist
		self.labels = labelList		
			  
			
	def __getitem__(self,index):
		image = Image.open(self.files[index])
		img_tmp = (numpy.array(image))
		if(img_tmp.ndim == 2):
			img_tmp = img_tmp.reshape((img_tmp.shape[0], img_tmp.shape[1], 1))
			img_tmp = numpy.concatenate([img_tmp, img_tmp, img_tmp], axis=2)

		image = Image.fromarray(img_tmp)
		if(self.transform):
			image = self.transform(image)
		
		label = self.labels[index]

		
		return image, label

	def __len__(self):
		return len(self.files)
			

