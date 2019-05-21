# -*- coding: utf-8 -*-
#This file is a simple neural network LeNet

__author = 'Lizzie'

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()	#???
		'''
		nn.Conv2d(x, y, z) >> x input channels, y output channels, z*z squre convolution kernel
		nn.Linear(m, n) >> an affine operation on vector m,n,b & matrix M, i.e y = Wx + b 
		'''
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)	# 16*5*5 regulate the input size to be 32*32
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		'''
		F.max_pool2d(N1, (w, w)) >> max pooling network N1 over a (w, w) window
		p.s when window is a squre like (w, w), u can use w instead of (w, w)
		'''
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))	# self.subs1 = x
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)	# self.subs2 = x
		x = x.view(-1, self.num_flat_features(x))	# define later
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


net = Net()

step = 4
if step == 1:
	'''
	Define the network
	'''
	params = list(net.parameters())
	'''
	params has 10 items:
	0.conv1's .weight
	1.subsampling1's .weight
	2.conv2's .weight
	3.subsampling's .weight
	4.fc1's .weight(or .bias)
	5.fc1's .weight(or .bias)
	6.fc2's .weight(or .bias)
	7.fc2's .weight(or .bias)
	8.fc3's .weight(or .bias)
	9.fc3's .weight(or .bias)
	'''
	print(len(params)) 
	print(params[0].size())

elif step == 2:
	'''
	Process inputs and call backward
	'''
	input = torch.randn(1, 1, 32, 32)
	out = net(input)
	print(out)
	net.zero_grad()
	out.backward(torch.randn(1,10))

elif step == 3 :
	'''
	Loss Function
	'''
	input = torch.randn(1, 1, 32, 32)
	output = net(input)
	target = torch.randn(10)
	target = target.view(1, -1)	# make it as the same shape as output
	criterion = nn.MSELoss()

	loss = criterion(output, target)
	print(loss)
	print(loss.grad_fn)
	print(loss.grad_fn.next_functions[0][0])
	print(loss.grad_fn.next_functions[0][0].next_functions[0][0])

elif step == 4:
	'''
	Backprop
	'''
	input = torch.randn(1, 1, 32, 32)
	output = net(input)
	target = torch.randn(10)
	target = target.view(1, -1)	# make it as the same shape as output
	criterion = nn.MSELoss()
	loss = criterion(output, target)

	net.zero_grad()

	print('conv1.bias.grad before backward')
	print(net.conv1.bias.grad)	#??? why None

	loss.backward()

	print('conv1.bias.grad after backward')
	print(net.conv1.bias.grad)

	'''
	Update weights
	'''
	learning_rate = 0.01
	for f in net.parameters():
		f.data.sub_(f.grad.data*learning_rate)