# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 20:24:31 2019

@author: Flo
"""

import compfy
import optionsData
import rohling

import numpy as np
from pandas_datareader import data
import arrow
from tqdm import tqdm
from time import time
import datetime

end = arrow.utcnow()
start = end.shift(years = -2).date()
end = end.date()

def it(t):
    return int(t/h)

class stock:
    
    def __init__(self, symbol, coeff):
        self.symbol = symbol
        self.S0 = optionsData.getCurrentPrice(symbol)
        self.v0, self.vq, self.rho, self.kap, self.sigm = np.loadtxt("theta"+symbol+".csv", delimiter = ";")
        self.coeff = coeff
        
        table = data.DataReader(symbol, "yahoo", start, end)
        self.history = (np.array(table["Open"])+np.array(table["Close"]))/2
        self.V = None
        self.S = None
        
    def aV(self, t, v):
        return self.kap*(self.vq - v)
    
    def bV(self, t, v):
        return self.sigm*np.sqrt(v)
    
    def aS(self, t, s):
        return mü * s
    
    def bS(self, t, s):
        return np.sqrt(self.V[it(t)])
        
def CallOnMax(stocks):
    return max(max([s.coeff * s.S[-1] for s in stocks]), 0)

def CallOnMin(stocks):
    return max(min([s.coeff * s.S[-1] for s in stocks]), 0)

def Exchange(stocks):
    return max(stocks[0].coeff * stocks[0].S[-1] - stocks[1].coeff * stocks[1].S[-1], 0)

"""
n = 5
symbols = ["IBM", "INTC", "NVDA", "GOOG", "AAPL"]
coeffs = [9, 22.5, 7.3, 1, 6.4]
"""

n = 2
symbols = ["PEP", "KO"]
coeffs = [1, 2.3]

now = time()
expdates = ["2019-02-01", "2019-02-15", "2019-06-21"]
Ts = [(datetime.datetime.strptime(expdate, "%Y-%m-%d").timestamp()-now)/60/60/24/356 for expdate in expdates]

T = Ts[0]

t = 76
mü = 0.01

N = 1000 #steps per simulation
M = 1000 #num of simulations

h = T/(N-1)

stocks = [stock(s, c) for s, c in zip(symbols, coeffs)]

sigma = np.eye(2*n)
for i in range(n):
    sigma[2*i, 2*i + 1] = stocks[i].rho
    sigma[2*i + 1, 2*i] = stocks[i].rho
    for j in range(n):
        if i < j:
            args = (stocks[i].history, stocks[j].history, t)
            sigma[2*i, 2*j] = rohling.rhoSS(*args)
            sigma[2*i, 2*j + 1] = rohling.rhoSV(*args)
            sigma[2*i + 1, 2*j] = rohling.rhoVS(*args)
            sigma[2*i + 1, 2*j + 1] = rohling.rhoVV(*args)
            sigma[2*j, 2*i] = sigma[2*i, 2*j]
            sigma[2*j + 1, 2*i] = sigma[2*i, 2*j + 1]
            sigma[2*j, 2*i + 1] = sigma[2*i + 1, 2*j]
            sigma[2*j + 1, 2*i + 1] = sigma[2*i + 1, 2*j + 1]
            
assert (sigma == sigma.T).all()
#np.linalg.cholesky(sigma)

dW = np.sqrt(h) * np.random.multivariate_normal(np.zeros(2 * n), sigma, (M, N))
for i in range(n):
    stocks[i].dWS = dW[:,:,2*i]
    stocks[i].dWV = dW[:,:,2*i + 1]

payoff = []
for i in tqdm(range(M)):
    for s in stocks:
        s.V, _ = compfy.EulerSDE(s.aV, s.bV, s.v0, T=T, N=N, dW = s.dWV[i], mode = "absolute")
        s.S, _ = compfy.EulerSDE(s.aS, s.bS, s.S0, T=T, N=N, dW = s.dWS[i])
    payoff.append(Exchange(stocks))
    
print(np.average(payoff))
    











