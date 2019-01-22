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
import matplotlib.pyplot as plt

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
        self.V = []
        self.S = []
        
    def aV(self, t, v):
        return self.kap*(self.vq - v)
    
    def bV(self, t, v):
        return self.sigm*np.sqrt(v)
    
    def aS(self, t, s):
        return mü * s
    
    def bS(self, t, s):
        return np.sqrt(self.V[-1][it(t)]) * s
    
    def adjustedEndValues(self):
        return [s[-1] * self.coeff for s in self.S]
    
    def reset(self):
        self.V = []
        self.S = []
        
        
def CallOnMax(stocks, K, T):
    #return max(max([s.coeff * s.S[-1] for s in stocks]), 0)
    payoff = []
    for m in range(M):
        payoff.append(np.exp(-mü*T) * max(max([s.S[m][-1]*s.coeff for s in stocks[:-1]]) - K, 0))
    return np.average(payoff)

def CallOnMin(stocks, K, T):
    payoff = []
    for m in range(M):
        payoff.append(np.exp(-mü*T) * max(min([s.S[m][-1]*s.coeff for s in stocks[:-1]]) - K, 0))
    return np.average(payoff)

def ExchangeWithMax(stocks, T):
    payoff=[]
    for m in range(M):
        payoff.append(np.exp(-mü*T)*max(max([s.S[m][-1]*s.coeff for s in stocks[:-1]])-stocks[-1].S[m][-1]*stocks[-1].coeff, 0))
    return np.average(payoff)

def ExchangeWithMin(stocks, T):
    payoff=[]
    for m in range(M):
        payoff.append(np.exp(-mü*T)*max(min([s.S[m][-1]*s.coeff for s in stocks[:-1]])-stocks[-1].S[m][-1]*stocks[-1].coeff, 0))
    return np.average(payoff)

def Exchange(stocks, T, reverse = False):
    if reverse:
        stocks.reverse()
    payoff = []
    for m in range(M):
        payoff.append(np.exp(-mü*T)*max(stocks[0].coeff * stocks[0].S[m][-1] - stocks[1].coeff * stocks[1].S[m][-1], 0))
    return np.average(payoff)


bsp = 2

if bsp == 2:
    n = 6
    symbols = ["IBM", "INTC", "NVDA", "GOOG", "AAPL", "XLK"]
    coeffs = [9, 22.5, 7.3, 1, 6.4, 16.5]
else:
    n = 2
    symbols = ["PEP", "KO"]
    coeffs = [1, 2.3]
    
"""
n = 1
symbols = ["KO"]
coeffs = [1]
"""

t = 25
mü = 0.0025

N = 1000 #steps per simulation
M = 1000 #num of simulations


stocks = [stock(s, c) for s, c in zip(symbols, coeffs)]
"""
#stocks.reverse()
for ka in range(100):
    t=ka*5+3
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
    print(t,min(np.linalg.eigvals(sigma)))
"""
minimierer=[1,-3]

for ka in tqdm(range(5,390)):
    sigma = rohling.CorrMatrix(stocks, ka)
    #np.linalg.eigvals(sigma)
    
    for i in range(n):
        sigma[2*i, 2*i + 1] = stocks[i].rho
        sigma[2*i + 1, 2*i] = stocks[i].rho
    if min(np.linalg.eigvals(sigma))>minimierer[1]:
        minimierer=[ka,min(np.linalg.eigvals(sigma))]

    #print(ka,min(np.linalg.eigvals(sigma)))
    
sigma=rohling.CorrMatrix(stocks,minimierer[0])
for i in range(n):
        sigma[2*i, 2*i + 1] = stocks[i].rho
        sigma[2*i + 1, 2*i] = stocks[i].rho
sigma=(sigma-min(minimierer[1]-0.00001,0)*np.eye(2*n))/(1-min(minimierer[1]-0.00001,0))

now = time()
expdates = ["2019-02-01", "2019-02-15", "2019-06-21"]
Ts = [(datetime.datetime.strptime(expdate, "%Y-%m-%d").timestamp()-now)/60/60/24/356 for expdate in expdates]

if bsp == 2:
    Ks = [1050, 1060, 1072, 1120]
    
    KTmax = dict()
    KTmin = dict()
    ETmax = dict()
    ETmin = dict()
else:
    E1 = dict()
    E2 = dict()


for T in Ts:
    
    h = T/(N-1)
    
    dW = np.sqrt(h) * np.random.multivariate_normal(np.zeros(2 * n), sigma, (M, N))
    for i in range(n):
        stocks[i].dWS = dW[:,:,2*i]
        stocks[i].dWV = dW[:,:,2*i + 1]
    
    
    for i in tqdm(range(M)):
        for s in stocks:
            s.V.append(compfy.EulerSDE(s.aV, s.bV, s.v0, T=T, N=N, dW = s.dWV[i], mode = "positive")[0])
            s.S.append(compfy.EulerSDE(s.aS, s.bS, s.S0, T=T, N=N, dW = s.dWS[i])[0])
    
    if bsp == 2:
        for K in Ks:
            KTmax[(K, T)] = CallOnMax(stocks, K, T)
            KTmin[(K, T)] = CallOnMin(stocks, K, T)
        ETmax[T] = ExchangeWithMax(stocks, T)
        ETmin[T] = ExchangeWithMin(stocks, T)
    else:
        E1[T] = Exchange(stocks, T)
        E2[T] = Exchange(stocks, T, reverse = True)
        
    for s in stocks:
        s.reset()



if bsp == 2:
    print("\nCall on Max:")
    print("K\T ", *expdates)
    for K in Ks:
        string = str(K)+" "
        for T in Ts:
            string += "{P:9.2f}$ ".format(P = KTmax[(K,T)])
        print(string)
    
    print("\nCall on Min:")
    print("K\T ", *expdates)
    for K in Ks:
        string = str(K)+" "
        for T in Ts:
            string += "{P:9.2f}$ ".format(P = KTmin[(K,T)])
        print(string)
    
    print("\nExchange with max:")
    print(*expdates)
    string = ""
    for T in Ts:
        string += "{P:9.2f}$ ".format(P = ETmax[T])
    print(string)
    
    print("\nExchange with min:")
    print(*expdates)
    string = ""
    for T in Ts:
        string += "{P:9.2f}$ ".format(P = ETmin[T])
    print(string)
    
else:
    print("\nExchange PEP for 2.3*KO:")
    print(*expdates)
    string = ""
    for T in Ts:
        string += "{P:9.2f}$ ".format(P = E1[T])
    print(string)
    
    print("\nExchange 2.3*KO for PEP:")
    print(*expdates)
    string = ""
    for T in Ts:
        string += "{P:9.2f}$ ".format(P = E2[T])
    print(string)
    

h = T/(N-1)
dW = np.sqrt(h) * np.random.multivariate_normal(np.zeros(2 * n), sigma, (M, N))
for i in range(n):
    stocks[i].dWS = dW[:,:,2*i]
    stocks[i].dWV = dW[:,:,2*i + 1]

for s in stocks:
    s.reset()
    s.V.append(compfy.EulerSDE(s.aV, s.bV, s.v0, T=T, N=N, dW = s.dWV[i], mode = "positive")[0])
    s.S.append(compfy.EulerSDE(s.aS, s.bS, s.S0, T=T, N=N, dW = s.dWS[i])[0])

for s in stocks:
    plt.plot(s.S[0]*s.coeff)
plt.legend(symbols)



