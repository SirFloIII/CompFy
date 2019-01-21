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
stock = "PEP"


#get time from now and exactly two years in the past
end = arrow.utcnow()
start = end.shift(years = -2).date()
end = end.date()

Pepsi = data.DataReader(stock, "yahoo", start, end)
Cola=data.DataReader("KO","yahoo",start,end)
Pepsi=np.array((np.array(Pepsi["Open"])+np.array(Pepsi["Close"]))/2)
Cola=np.array((np.array(Cola["Open"])+np.array(Cola["Close"]))/2)
t=42

def rhoSS(Stock1,Stock2,t):
    erg=np.corrcoef(Stock1,Stock2)
    return erg[0,1]
    
def rhoSV(Stock1,Stock2,t):
    V=StocktoVol(Stock2,t)
    
    erg=np.corrcoef(Stock1[-len(V):],V)
    return erg[0,1]
    
def rhoVV(Stock1,Stock2,t):
    V1=StocktoVol(Stock1,t)
    V2=StocktoVol(Stock2,t)
    erg=np.corrcoef(V1,V2)
    return erg[0,1]
    

def StocktoVol(Stock,t):
    erg=[]
    for i in range(len(Stock)-t):
        a=Stock[i:i+t]
        a=np.log(a[1:]/a[:-1])
        erg.append(np.std(a))
    return erg


p1=rhoSS(Pepsi,Cola,t)
p2=rhoSV(Cola,Pepsi,t)
p3=rhoSV(Pepsi,Cola,t)
p4=rhoVV(Cola,Pepsi,t)





