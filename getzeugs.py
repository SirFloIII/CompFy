# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:00:43 2019

@author: Thomas
"""

import optionsData as op
import numpy

Cola=op.getOptionDataFromYahoo("KO")
Pepsi=op.getOptionDataFromYahoo("PEP")
IBM=op.getOptionDataFromYahoo("IBM")
Intel=op.getOptionDataFromYahoo("INTC")
NVIDIA=op.getCurrentPrice("NVDA")
Google=op.getCurrentPrice("GOOG")
Apple=op.getOptionDataFromYahoo("AAPL")
XLK=op.getOptionDataFromYahoo("XLK")


numpy.savetxt('Cola.csv', Cola, fmt='%f', delimiter=';')
numpy.savetxt('Pepsi.csv',Pepsi,fmt='%f',delimiter=';')
numpy.savetxt('IBM.csv',IBM,fmt='%f',delimiter=';')
numpy.savetxt('Intel.csv',Intel,,fmt='%f',delimiter=';')
numpy.savetxt('NVDIA.csv',NVIDIA,fmt='%f',delimiter=';')




