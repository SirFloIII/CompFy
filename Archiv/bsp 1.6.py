# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:45:34 2018

@author: Flo
"""

import numpy as np

N = 10000000 #<- fuck the hint, with loops this would not be feasable

t = np.linspace(0,1,num = N+1)

Z = np.random.normal(size = N+1)
Z[0] = 0

B = np.cumsum(np.sqrt(1/N)*Z)

from matplotlib import pyplot as plt
plt.plot(t,B)