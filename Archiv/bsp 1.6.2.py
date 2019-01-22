# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 23:57:54 2018

@author: Flo
"""

import numpy as np

"""
lets do it again with random nonuniform timesteps
"""

N = 1000

t = np.random.uniform(size = N)
t = np.concatenate((t, [0,1]))
t.sort()

Z = np.random.normal(size = N)
Z = np.concatenate(([0], Z))

B = np.cumsum(np.sqrt(t[1:] - t[:-1])*Z)

from matplotlib import pyplot as plt
plt.plot(t[:-1],B)