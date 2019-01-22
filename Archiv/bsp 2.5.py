# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 21:48:41 2018

@author: Flo
"""

import numpy as np
import compfy

#Itô-Prozess mit a = 0 und b = sin(t)
#dX = sin(t)*dW
for _ in range(1000):
    compfy.plotItô(0, np.sin)