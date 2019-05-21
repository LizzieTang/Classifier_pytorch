# -*- coding: utf-8 -*-
# Define a CNN for image classification

__author = 'Lizzie'


import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()	#???
		'''
		nn.Conv2d(x, y, z) >> x input channels, y output channels, z*z squre convolution kernel
		nn.Linear(m, n) >> an affine operation on vector m,n,b & matrix M, i.e y = Wx + b 
		'''
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(16*5*5, 120)	# 16*5*5 regulate the input size to be 32*32
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		'''
		F.max_pool2d(N1, (w, w)) >> max pooling network N1 over a (w, w) window
		p.s when window is a squre like (w, w), u can use w instead of (w, w)
		'''
		x = self.pool1(F.relu(self.conv1(x)))	# self.subs1 = x
		x = self.pool2(F.relu(self.conv2(x)))	# self.subs2 = x
		# x = x.view(-1, self.num_flat_features(x))	# define later
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	# def num_flat_features(self, x):
	# 	size = x.size()[1:]
	# 	num_features = 1
	# 	for s in size:
	# 		num_features *= s
	# 	return num_features



net = Net()