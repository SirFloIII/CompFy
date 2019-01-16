# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 01:04:35 2019

@author: Flo
"""

import numpy as np
from datetime import datetime

def timediff(a, b):
    date_format = "%m/%d/%y"
    a = datetime.strptime(a, date_format)
    b = datetime.strptime(b, date_format)
    delta = b - a
    return delta.days

def convertFileToKTPMatrix(filename = "options-screener-01-16-2019.csv", day = "01/16/19", symbol = None, optiontype = "Call"):
    """
    reads a barchart csv to create the desired (K|T|Price)-Matrix (T is in days)
    filename ... full filename including .csv
    day ... the day of the prices, in "mm/dd/yy" format
    
    symbol ... if none, most common symbol in log is taken, else specified symbol is taken
    optiontype ... "Call" or "Put"
    
    """
    
    with open(filename, "r") as f: #using the with statment allows us to not care about closing the file again
        f.readline() #first line is legend, so wie read it once so the rest is only data
        lines = f.readlines() #this is a list of strings
    lines.pop() #the last line is some "downloaded form barchart" bullshit, so we pop it away
    
    lines = [line.split(",") for line in lines] #each line is a list now, listing:
    #Symbol, Price, Type, Strike, DTE, "Exp Date", Bid, Midpoint, Ask, Last, Volume, "Open Int", Vol/OI, IV, Time
    
    if symbol == None:
        symbols = list(set([line[0] for line in lines])) #converting to set and back gets rid of duplicates
        counts = [len([line for line in lines if line[0] == symbol and line[2] == "Call"]) for symbol in symbols]
        symbol = symbols[np.argmax(counts)] #we will analyse the symbol with the most occurences
    
    #building the wanted (K|T|Callpreis) Matrix:
    CallOptions = [[line[4], timediff(day, line[5]), line[7]] for line in lines if line[0] == symbol and line[2] == "Call"]
    
    return np.array(CallOptions, dtype = np.float)

if __name__ == "__main__":
    
    C = convertFileToKTPMatrix()