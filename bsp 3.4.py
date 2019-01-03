# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 20:22:04 2018

@author: Flo
"""

import numpy as np
import compfy
import matplotlib.pyplot as plt


if False:
    r = 0.0328
    T = 0.211
    S0 = 5290.36

    K = np.array((6000, 6200, 6300, 6350, 6400, 6600, 6800))
    C = np.array((80.2, 47.1, 35.9, 31.9, 27.7, 16.6, 11.4))
else:
    Symbol = "SNAP"
    
    r = 0.0
    T = 2/252
    S0 = 6.59
    
    K = np.array(( 6.5,  7.0,  7.5,  8.0,  8.5,  9.0, 10.5))
    C = np.array((0.63, 0.43, 0.27, 0.18, 0.09, 0.07, 0.03))
    
sL = np.ones(K.shape) * 1e-2
sU = np.ones(K.shape) * 5

assert np.all(C > compfy.BlackScholesCall(S0, K, r, sL, 0, T))
assert np.all(C < compfy.BlackScholesCall(S0, K, r, sU, 0, T))

for _ in range(20):
    sA = (sL + sU)/2
    tooLow = C > compfy.BlackScholesCall(S0, K, r, sA, 0, T)
    tooHigh = np.logical_not(tooLow)
    
    sL[tooLow] = sA[tooLow]
    sU[tooHigh] = sA[tooHigh]
    
plt.plot(K, sA, marker = "o")