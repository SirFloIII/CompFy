# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 21:50:01 2018

@author: Flo
"""

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from tqdm import tqdm

def solveDiffusionEq(v, u0, u1, a, b, T, theta, N, M, loadingBar = True):
    """
    solves uxx(x,t) = ut(x,t) in (a,b)x(0,T)
    for u(a,t) = u0(t)
    u(b,t) = u1(t)
    u(x,0) = v(x)
    """
    
    import scipy.sparse as sp
    
    #h = (b-a)/N
    #tau = T/M
    gamma = T /M /(b-a)**2 * N**2 # = tau/h²
    
    x = np.linspace(a, b, num = N+1)
    t = np.linspace(0, T, num = M+1)
    
    w = np.zeros((N+1, M+1))
    w[:,0] = v(x)
    w[0,:] = u0(t)
    w[N,:] = u1(t)
    
    """
    (6.4)
    
    (w[i,j+1] - w[i,j]) = gamma * ((1-theta)*(w[i+1,j] - 2*w[i,j] + w[i-1,j]) + theta*(w[i+1,j+1] - 2*w[i,j+1] + w[i-1,j+1]))
    """
    
    A = sp.diags([-gamma * theta  * np.ones(N-2), ( 2*gamma *  theta +1)*np.ones(N-1), -gamma * theta  * np.ones(N-2)], [-1, 0, 1]).toarray()
    B = sp.diags([gamma*(1-theta) * np.ones(N-2), (-2*gamma*(1-theta)+1)*np.ones(N-1), gamma*(1-theta) * np.ones(N-2)], [-1, 0, 1]).toarray()
    
    d = np.zeros((N+1,M))
    d[1]   = gamma*((1-theta)*u0(t[:-1]) + theta*u0(t[1:]))
    d[N-1] = gamma*((1-theta)*u1(t[:-1]) + theta*u1(t[1:]))
    
    for j in tqdm(range(M)) if loadingBar else range(M):
        
        #A*w_j+1 = B*w_j + d_j
        
        w[1:N, j+1] = np.linalg.solve(A, B.dot(w[1:N, j]) + d[1:N, j])
    
    return w, x, t

def EulerSDE(a, b, S0, T = 1, N = 1000, dW = None, mode = "standard"):
    """
    dSt = a(t,St) * dt + b(t,St) * dW
    a, b must be functions of t and S in this order.
    for a, b functions of t or const use Itô(...)
    
    if dW is given, it is used, else generate a new one
    assert dW.size >= N
    """
    
    dt = T/N
    if dW is None:
        dW = np.random.normal(scale = np.sqrt(dt), size = N)
    else:
        assert dW.size >= N
    
    S = np.zeros(N)
    S[0] = S0
    
    t = np.linspace(0, T, num = N)
    
    for i in range(N-1):
        S[i+1] = S[i] + a(t[i], S[i])*dt + b(t[i], S[i])*dW[i]
        if mode == "standard":
            pass
        elif mode == "positive":
            S[i+1] = max(0, S[i+1])
        elif mode == "absolute":
            S[i+1] = abs(S[i+1])
    
    return S, t
    
def wiener(N, T = 1):
    Z = np.random.normal(size = N)
    Z[0] = 0
    
    return np.cumsum(np.sqrt(1/N * T)*Z)

def Itô(a, b, T = 1, N = 1000):
    
    if isinstance(a, float) or isinstance(a, int):
        av = a
        a = lambda x:av
    if isinstance(b, float) or isinstance(b, int):
        bv = b
        b = lambda x:bv
    
    t = np.linspace(0, T, N)
    
    W = wiener(N+1, T = T)
    
    dX = a(t) * T * 1/N + b(t) * (W[1:] - W[:-1])
    
    X = np.cumsum(dX)
    
    return X, t

def plotItô(a, b, T = 1, N = 1000):
    X, t = Itô(a, b, T = T, N = N)
    plt.plot(t,X)

def d1(S, K, r, sigma, t, T):
    return (np.log(S/K) + (r + sigma**2/2)*(T-t)) / (sigma * (np.sqrt(T-t)))

def d2(S, K, r, sigma, t, T):
    return (np.log(S/K) + (r - sigma**2/2)*(T-t)) / (sigma * np.sqrt((T-t)))

def Phi(x):
    return norm.cdf(x)
    
def BlackScholesCall(S, K, r, sigma, t, T):
    return S * Phi(d1(S, K, r, sigma, t, T)) - K * np.exp(-r*(T-t)) * Phi(d2(S, K, r, sigma, t, T))

def BlackScholesPut(S, K, r, sigma, t, T):
    return K*np.exp(r*(T-t))*Phi(-d2(S, K, r, sigma, t, T)) - S*Phi(-d1(S, K, r, sigma, t, T))
    
if __name__ == "__main__":
    S = np.linspace(0.01, 2)
    K = 1
    r = 0.1
    sigma = 1
    T = 1
    for t in np.linspace(0, T, num = 20, endpoint = False):
        plt.plot(S, BlackScholesCall(S, K, r, sigma, t, T))