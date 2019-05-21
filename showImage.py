# -*- coding: utf-8 -*-
# show images for fun ???

__author = 'Lizzie'

import matplotlib.pyplot as plt
import numpy as np
from dataloader import trainloader, classes
import torchvision as tv

'''
show images
'''
def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(tv.utils.make_grid(images))

print('GroundTruth:',' '.join('%5s' % classes[labels[j]] for j in range(4)))