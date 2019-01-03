# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 15:52:58 2018

@author: Flo
"""

import numpy as np
import compfy
from tqdm import tqdm

"""
dSt = rSt dt + sigma*St dWt
"""

S0 = 10
K = 12
r = 0.04
sigma = 0.4
T = 2
H = 13

a = lambda t, S: r*S
b = lambda t, S: sigma*S

m = 1000
N = 10000

f = lambda y : compfy.EulerSDE(a, b, S0, T, N)[0]

S = list(map(f, tqdm(range(m))))

Payoff = list(map(lambda X : np.exp(-r*T) * max(K-X[-1], 0) * np.all(X < H), S))

E = np.average(Payoff)

print("Erwartungswert:", E)

V = np.sum((Payoff - E)**2) / N

print("Varianz:", V)