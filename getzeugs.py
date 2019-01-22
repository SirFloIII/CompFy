# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:00:43 2019

@author: Thomas
"""

import optionsData as op
import numpy

symbols = ["PEP", "KO", "IBM", "INTC", "NVDA", "GOOG", "AAPL","XLK"]

for s in symbols:
    data = op.getOptionDataFromYahoo(s)
    numpy.savetxt(s+".csv", data, fmt = "%f", delimiter = ";")
