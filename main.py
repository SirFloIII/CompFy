# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 16:11:34 2019

@author: Flo
"""

import optionsData
import Heston_calibration
import eulerMarijuana

import numpy as np

dataPEP = optionsData.getOptionDataFromYahoo("PEP")
dataKO = optionsData.getOptionDataFromYahoo("KO")

mü = 0.01
theta=[0.08,0.1,-0.7,3,0.25] #startwerte

PEP0 = optionsData.getCurrentPrice("PEP")
thetaPEP, _ = Heston_calibration.LevMarquCali(theta, PEP0, mü, dataPEP)

KO0 = optionsData.getCurrentPrice("KO")
thetaKO, _ = Heston_calibration.LevMarquCali(theta, KO0, mü, dataKO)

#theta=(v0,vbar,rho,kap,sigma)