# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:11:34 2019

@author: Flo
"""

import optionsData
import Heston_calibration
import eulerMarijuana
import rohling
from tqdm import tqdm


import numpy as np
import datetime
from time import time
"""
dataPEP = optionsData.getOptionDataFromYahoo("PEP")
dataKO = optionsData.getOptionDataFromYahoo("KO")
"""
mü = 0.01
theta=[0.08,0.1,-0.7,3,0.25] #startwerte


"""
PEP0 = optionsData.getCurrentPrice("PEP")
KO0 = optionsData.getCurrentPrice("KO")

thetaPEP, _ = Heston_calibration.LevMarquCali(theta, PEP0, mü, dataPEP)
thetaKO, _ = Heston_calibration.LevMarquCali(theta, KO0, mü, dataKO)

#theta=(v0,vbar,rho,kap,sigma)

v0p, vqp, rhop, kapp, sigmap = thetaPEP
v0c, vqc, rhoc, kapc, sigmac = thetaKO
"""

KO0 = 47.61
v0c = 0.684
vqc = 0.593
rhoc = -0.431
kapc = 3.03
sigmac = 0.894

PEP0 = 110.07
v0p = 0.470
vqp = 0.351
rhop = -0.39
kapp = 3.074
sigmap = 0.168
"""
KO0 = 40
v0c = 0.2
vqc = 0.2
rhoc = -0.7
kapc = 3
sigmac = 0.3

PEP0 = 110
v0p = 0.2
vqp = 0.2
rhop = -0.7
kapp = 3
sigmap = 0.3
"""
N = 1000
#expiration dates: 1.2.2019, 15.2.2019, 21.6.2019
now = time()
expdates = ["2019-02-01", "2019-02-15", "2019-06-21"]
Ts = [(datetime.datetime.strptime(expdate, "%Y-%m-%d").timestamp()-now)/60/60/24/356 for expdate in expdates]

p1, p2, p3, p4 = rohling.getps()
#p1, p2, p3, p4 = 0,0,0,0

payoff = []
for _ in tqdm(range(1000)):
    PEP, KO = eulerMarijuana.simulate(PEP0, KO0, v0p, v0c, mü, vqp, vqc, sigmap, sigmac, kapp, kapc, rhop, rhoc, p1, p2, p3, p4, N, Ts[0])
    #payoff.append(max(- 2.3*KO[-1] + PEP[-1], 0))
    payoff.append(max(PEP[-1] - 110, 0))

print(np.average(payoff))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    