# -*- coding: utf-8 -*-
# Define a loss function

__author = 'Lizzie'

import torch.optim as optim
import torch.nn as nn
from CNNClassifier import net


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), 	# update weights
					  lr = 0.001, 
					  momentum = 0.9)

