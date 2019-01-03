# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 23:04:26 2018

@author: Flo
"""

#import numpy as np
import compfy

#Itô-Prozess mit a = 5 und b = 3
#dX = 5*dt + 3*dW
for _ in range(1000):
    compfy.plotItô(5, 3)