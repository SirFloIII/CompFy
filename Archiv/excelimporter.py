# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 15:50:37 2019

@author: Thomas
"""

import numpy as np
#import pandas as pd
import pandas

filename="Pepsi.csv"

with open(filename, "r") as f: #using the with statment allows us to not care about closing the file again
    f.readline() #first line is legend, so wie read it once so the rest is only data
    lines = f.readlines() #this is a list of strings
 
lines = [line.split(";") for line in lines]    


daten= pandas.read_csv(filename,sep=";",header=None,decimal=",")

daten=np.array(daten)
