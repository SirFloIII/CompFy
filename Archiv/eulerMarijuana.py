# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:34:30 2019

@author: Flo
"""

import numpy as np
import compfy
#from tqdm import tqdm

def simulate(PEP0, KO0, vp0, vc0, mü, vqp, vqc, sigmap, sigmac, Kp, Kc, pp, pc, p1, p2, p3, p4, N, T):
    h = T/(N-1)
    
    #generierung der korrelierten normalverteilungen
    var = np.array([[1, pp, p1, p2],
                    [pp, 1, p3, p4],
                    [p1, p3, 1, pc],
                    [p2, p4, pc, 1]])
    
    #dW = np.random.normal(size = (4,N))
    #dW = np.sqrt(h) * np.linalg.cholesky(var).dot(dW)
    
    dW = np.sqrt(h) * np.random.multivariate_normal(np.zeros(4), var, N).T
    
    #defining a and b for the volatiliy
    avp = lambda t, v : Kp*(vqp - v)
    avc = lambda t, v : Kc*(vqc - v)
    
    bvp = lambda t, v : sigmap*np.sqrt(v)
    bvc = lambda t, v : sigmac*np.sqrt(v)
    
    vp, _ = compfy.EulerSDE(avp, bvp, vp0, T=T, N=N, dW = dW[1], mode = "absolute")
    vc, _ = compfy.EulerSDE(avc, bvc, vc0, T=T, N=N, dW = dW[3], mode = "absolute")
    
    #defing a and b for the stocks themself
    aPEP = lambda t, S : mü * S
    aKO = lambda t, S : mü * S
    
    i = lambda t : int(t/h)
    bPEP = lambda t, S : np.sqrt(vp[i(t)])
    bKO = lambda t, S : np.sqrt(vc[i(t)])
    
    PEP, _ = compfy.EulerSDE(aPEP, bPEP, PEP0, T=T, N=N, dW = dW[0])
    KO, _ = compfy.EulerSDE(aKO, bKO, KO0, T=T, N=N, dW = dW[2])
    
    return PEP, KO

if __name__ == "__main__":
        
    #startwerte
    PEP0 = 140
    KO0 = 50 
    vp0 = 10
    vc0 = 10
    
    #konstanten
    mü = 0.01
    vqp = 10
    vqc = 11
    sigmap = 3
    sigmac = 2
    Kp = 0.1
    Kc = 0.1
    
    #korrelationskoeffizenten
    pp = 0.5
    pc = 1
    p1, p2, p3, p4 = [1,2,3,4]
    
    #simulationsdetails
    N = 1000
    T = 20

    PEP, KO = simulate(PEP0, KO0, vp0, vc0, mü, vqp, vqc, sigmap, sigmac, Kp, Kc, pp, pc, p1, p2, p3, p4, N, T)


