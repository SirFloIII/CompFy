# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:11:34 2019

@author: Flo
"""

import optionsData
import Heston_calibration
import eulerMarijuana

import numpy as np
import datetime
from time import time

dataPEP = optionsData.getOptionDataFromYahoo("PEP")
dataKO = optionsData.getOptionDataFromYahoo("KO")

mü = 0.01
theta=[0.08,0.1,-0.7,3,0.25] #startwerte

PEP0 = optionsData.getCurrentPrice("PEP")
thetaPEP, _ = Heston_calibration.LevMarquCali(theta, PEP0, mü, dataPEP)

KO0 = optionsData.getCurrentPrice("KO")
thetaKO, _ = Heston_calibration.LevMarquCali(theta, KO0, mü, dataKO)

#theta=(v0,vbar,rho,kap,sigma)

v0p, vqp, rhop, kapp, sigmap = thetaPEP
v0c, vqc, rhoc, kapc, sigmac = thetaKO

N = 1000
#expiration dates: 1.2.2019, 15.2.2019, 21.6.2019
now = time()
expdates = ["2019-02-01", "2019-02-15", "2019-06-21"]
Ts = [(datetime.datetime.strptime(expdate, "%Y-%m-%d").timestamp()-now)/60/60/24/356 for expdate in expdates]



PEP, KO = eulerMarijuana.simulate(PEP0, KO0, v0p, v0c, mü, vqp, vqc, sigmap, sigmac, kapp, kapc, rhop, rhoc, p1, p2, p3, p4, N, Ts[0])