# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 18:00:43 2019

@author: Thomas
"""

import optionsData as op
import numpy

Cola=op.getOptionDataFromYahoo("KO")
Pepsi=op.getOptionDataFromYahoo("PEP")


numpy.savetxt('Cola.csv', Cola, fmt='%f', delimiter=';')
numpy.savetxt('Pepsi.csv',Pepsi,fmt='%f',delimiter=';')





