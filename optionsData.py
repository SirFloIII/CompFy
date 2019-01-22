# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 01:04:35 2019

@author: Flo
"""

import numpy as np
from datetime import datetime
import requests
import json
from time import time


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

def getOptionDataFromYahoo(symbol, optiontype = "Call"):
    page = requests.get("https://query1.finance.yahoo.com/v7/finance/options/"+symbol)
    content = json.loads(page.text)
    
    K = []
    T = []
    P = []
    
    for exp in content["optionChain"]["result"][0]["expirationDates"]:
        page = requests.get("https://query1.finance.yahoo.com/v7/finance/options/"+symbol+"?date="+str(exp))
        content = json.loads(page.text)
        
        if optiontype == "Call":
            options = content["optionChain"]["result"][0]["options"][0]["calls"]
        elif optiontype == "Put":
            options = content["optionChain"]["result"][0]["options"][0]["puts"]
        else:
            assert optiontype == "Call" or optiontype == "Put"
            
        K += [option["strike"] for option in options]
        
        now = time()
        T += [(option["expiration"] - now)/60/60/24/356 for option in options]
        
        P += [(option["ask"]+option["bid"])/2 for option in options]
        
    KTP = np.array((K,T,P)).T
    
    return KTP

def getCurrentPrice(symbol):
    page = requests.get("https://query1.finance.yahoo.com/v7/finance/options/"+symbol)
    content = json.loads(page.text)
    
    quote = content["optionChain"]["result"][0]["quote"]
    
    return quote["regularMarketPrice"]
    
if __name__ == "__main__":
    
    #symbols = ["KO", "PEP", "IBM", "INTC", "NVDA", "GOOG", "AAPL"]
    symbols = ["XLK"]
    optiontype = "Call"
    
    KTP = [getOptionDataFromYahoo(symbol, optiontype = optiontype) for symbol in symbols]
    
    #C = convertFileToKTPMatrix()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    