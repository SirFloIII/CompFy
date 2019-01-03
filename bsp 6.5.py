# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:21:43 2018

@author: Flo'
"""

"""
Heston Modell:
dSt = mt*St*dt + sigmat*St*dWt

dUt = k(theta - Ut) dt + lambdat sqrt(Ut)*dW~t

mit Ut := sigmat**2
dW*dW~ = rho*dt
"""

import numpy as np
import compfy
from tqdm import tqdm
#from matplotlib import pyplot as plt

S0 = 100
K = 100
r = 0.05
lam = 0.3
theta = 0.04
rho = -0.5
kappa = 1.2
T = 1

N = 1000
m = 1000

value = []
for _ in tqdm(range(m)):
    
    dW1 = np.random.normal(scale = np.sqrt(T/N), size = N)
    dZ = np.random.normal(scale = np.sqrt(T/N), size = N)
    dW2 = rho*dW1 + np.sqrt(1-rho**2) * dZ
    
    #for U
    a1 = lambda t, U : kappa*(theta - U)
    b1 = lambda t, U : lam * np.sqrt(U)
    
    U, _ = compfy.EulerSDE(a1, b1, theta, T = T, N = N, dW = dW1, mode = "absolute")
    
    #for S
    a2 = lambda t, S : r * S
    b2 = lambda t, S : np.sqrt(U[int(t*N)] * S)
    
    S, _ = compfy.EulerSDE(a2, b2, S0, T = T, N = N, dW = dW2)
    
    value.append(S[-1])


payoff = list(map(lambda S : max(S-K, 0) * np.exp(-r*T), value))

V = np.average(payoff)

print("V =", V)