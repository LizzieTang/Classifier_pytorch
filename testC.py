# -*- coding: utf-8 -*-
# test the net

__author = 'Lizzie'


from dataloader import classes
from showImage import images
from trainC import net
import torch as t

outputs = net(images)

a, prediction = t.max(outputs, 1)

print('Predicted:', ' '.join('%5s' % classes[prediction[j]] for j in range(4)))
