# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:34:30 2019

@author: Flo
"""

import numpy as np
import compfy
#from tqdm import tqdm

#startwerte
PEP0 = 140
KO0 = 50 
vp0 = 10
vc0 = 10

#konstanten
müp = 1
müc = 1
vqp = 10
vqc = 11
sigmap = 3
sigmac = 2
K = 0.1

#korrelationskoeffizenten
pp = 0.5
pc = 1
p1, p2, p3, p4 = [1,2,3,4]

#simulationsdetails
N = 1000
T = 20
h = T/(N-1)

#generierung der korrelierten normalverteilungen
var = np.array([[1, pp, p1, p2],
                [pp, 1, p3, p4],
                [p1, p2, 1, pc],
                [p2, p4, pc, 1]])

dW = np.random.normal(size = (4,N))
dW = np.sqrt(h) * var.dot(dW)

#defining a and b for the variance
avp = lambda t, v : K*(vqp - v)
avc = lambda t, v : K*(vqc - v)

bvp = lambda t, v : sigmap*np.sqrt(v)
bvc = lambda t, v : sigmac*np.sqrt(v)

vp, _ = compfy.EulerSDE(avp, bvp, vp0, T=T, N=N, dW = dW[1], mode = "absolute")
vc, _ = compfy.EulerSDE(avc, bvc, vc0, T=T, N=N, dW = dW[3], mode = "absolute")

#defing a and b for the stocks themself
aPEP = lambda t, S : müp * S
aKO = lambda t, S : müc * S

i = lambda t : int(t/h)
bPEP = lambda t, S : np.sqrt(vp[i(t)])
bKO = lambda t, S : np.sqrt(vc[i(t)])

PEP, _ = compfy.EulerSDE(aPEP, bPEP, PEP0, T=T, N=N, dW = dW[0])
KO, _ = compfy.EulerSDE(aKO, bKO, KO0, T=T, N=N, dW = dW[2])


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    