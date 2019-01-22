# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:00:43 2019

@author: Thomas
"""

import optionsData as op
import numpy

symbols = ["PEP", "KO", "IBM", "INTC", "NVDA", "GOOG", "AAPL", "XLK"]

for s in symbols:
    data = op.getOptionDataFromYahoo(s)
    numpy.savetxt("KTP_"+s+".csv", data, fmt = "%f", delimiter = ";")
    
    S0 = [op.getCurrentPrice(s)]
    numpy.savetxt("S0_"+s+".csv", S0)