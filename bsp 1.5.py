# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 22:41:27 2018

@author: Flo
"""

import numpy as np
#import pandas as pd
from pandas_datareader import data
from matplotlib import pyplot as plt
import arrow

#which stock do we want?
stock = "MCD"


#get time from now and exactly two years in the past
end = arrow.utcnow()
start = end.shift(years = -2).date()
end = end.date()

panel_data = data.DataReader(stock, "yahoo", start, end)


#The manual way, using numpy:
opening = np.array(panel_data["Open"]) #<- we lose date information in this step, not recommended
mean = np.sum(opening)/np.size(opening)
var = np.sum((opening - mean)**2)/(np.size(opening)-1)
#plt.plot(opening)

#The lazy way, using pandas
mean2 = panel_data["Open"].mean()
var2 = panel_data["Open"].var()
plt.plot(panel_data["Open"])

#The really lazy way
print(panel_data["Open"].describe())
#sadly this is too lazy and it does not include the variance
#it includes the standard deviation, its square root. eh, close enough