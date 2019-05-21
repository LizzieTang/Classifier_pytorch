# -*- coding: utf-8 -*-
# train the classifier network

__author = 'Lizzie'

from dataloader import trainloader
from CNNClassifier import net
from Loss import criterion, optimizer


for epoch in range(2):	# loop over dataset 2 times
	
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		'''
		inputs data
		'''
		inputs, labels = data

		# reset parameter gradients
		optimizer.zero_grad()

		# forward
		outputs = net(inputs)

		# loss + backward
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print 
		running_loss += loss.item()
		if i % 2000 == 1999:	# print / 2000 mini-batches
			print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0


print('Finish Training')