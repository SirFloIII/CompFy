# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 16:01:24 2018

@author: Flo
"""

import numpy as np
import matplotlib.pyplot as plt
import compfy

"""
solves uxx(x,t) = ut(x,t) in (a,b)x(0,T)
for u(a,t) = u0(t)
    u(b,t) = u1(t)
    u(x,0) = v(x)
"""


v  = lambda x : np.sin(5 * np.pi * x)**2
u0 = lambda t : 0
u1 = lambda t : 0

a = 0
b = 1
T = 0.01

theta = 1/2

#spacesteps
N = 500
#timesteps
M = 500

w, x, t = compfy.solveDiffusionEq(v, u0, u1, a, b, T, theta, N, M)

plt.imshow(w, aspect = "auto")









