# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:00:56 2018

@author: Flo
"""

import compfy
from matplotlib import pyplot as plt

import numpy as np

"""
price european call with heat diffusion equation
"""

S0 = 10
K = 12
r = 0.04
sigma = 0.4
T = 2

"""
transformation according to lecture notes:
x = ln(S/K), tau = 1/2*sigma²*(T-t), v(x,tau) = V(S,t)/K, k = 2r/sigma²

v(x,tau) = exp(alpha*x + beta*tau)*u(x,tau)
with alpha = 1/2 * (1-k) and beta = -1/4 * (k+1)²

u_tau - u_xx = 0
u(x,0) = exp((k-1)x/2) * (e^x-1)^+
"""

k = 2*r/sigma**2

u0 = lambda x : np.exp((k-1)*x/2) * np.maximum((np.exp(x) - 1), 0)

theta = 1/2
N = 500
M = N

L = 100
uL = lambda t : 0
uU = lambda t : u0(L)

h = 2*L/N
ttau = T/M

uGrid, xGrid, tauGrid = compfy.solveDiffusionEq(u0, uL, uU, -L, L, 1/2*sigma**2 * T, theta, N, M)
def u(x, tau):
    i = int(np.floor((x-L)*h))
    alpha = x-L - i/h
    j = int(np.floor(tau*ttau))
    beta = tau - j/ttau
    
    return alpha * beta * uGrid[i,j] + (1-alpha) * beta * uGrid[i+1,j] + alpha * (1-beta) * uGrid[i, j+1] + (1-alpha)*(1-beta) * uGrid[i+1, j+1]

plt.imshow(uGrid, aspect = "auto")

alpha = 1/2 * (1-k)
beta = -1/4 * (k+1)**2

tau = lambda t : 1/2 * sigma**2 * (T-t)
x = lambda S : np.log(S/K)

v = lambda x, tau : np.exp(alpha*x + beta*tau) * u(x, tau)

V = lambda S, t : v(x(S), tau(t)) * K

print("Heatequation says: ", V(S0, 0))
print("BlSc-Formula says:", compfy.BlackScholesCall(S0, K, r, sigma, 0, T))