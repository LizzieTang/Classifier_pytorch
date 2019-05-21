# -*- coding: utf-8 -*-
# A simple image classifier
# use dataset CIFAR10

__author = 'Lizzie'

'''
1. Load and nomalize CIFAR10 training & test datasets via torchvision
2. Define a CNN
3. Define a loss function
4. Train the network on training data
5. Test the network on the test data
'''

import torch as t
import torchvision as tv
import torchvision.transforms as transforms


'''
{???}what is shuffle,
{???}train = False what would happen
'''
transform = transforms.Compose(
							    [transforms.ToTensor(),
								 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = tv.datasets.CIFAR10(root = './data',
										train = True,
										download = True,
										transform = transform)
trainloader = t.utils.data.DataLoader(trainset,
										  batch_size = 4,
										  shuffle = True,
										  num_workers = 2)
testset = tv.datasets.CIFAR10(root = './',
									   train = False,
									   download = True,
									   transform = transform)
testloader = t.utils.data.DataLoader(testset,
									 batch_size = 4,
									 shuffle = False,
									 num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
