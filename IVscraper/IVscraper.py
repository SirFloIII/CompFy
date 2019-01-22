# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:29:06 2019

@author: Flo
"""

import pyautogui as pag
import pytesseract


"""
hopefully reads marketchaeleon iv data via screenreading
use with 150% zoomlvl
"""



startx = 323
endx = 1660

y = 690

pag.PAUSE = 0.01

pag.hotkey("alt", "tab")

pag.moveTo(startx, y)


dates = []
values = []

for x in range(startx, endx):
    pag.moveTo(x, y)
    pos = pag.locateOnScreen("IV30.png", confidence = 0.8)
    if pos != None:
        datePic = pag.screenshot(region = (pos[0]-41, pos[1]-22, 90, 17))
        valuePic = pag.screenshot(region =(pos[0]+44, pos[1]   , 42, 16))
        
        dates.append(pytesseract.image_to_string(datePic))
        values.append(pytesseract.image_to_string(valuePic))


valuesf = []

for i in range(len(dates)-1):
    if dates[i] != dates[i+1] or values[i] != values[i+1]:
        valuesf.append(values[i])

for v in valuesf:
    try:
        float(v.replace(" ","").replace("A", ".4"))
    except:
        print(v)

valuesf = [float(v.replace(" ","").replace("A", ".4")) for v in valuesf if v != ""]

#with f as open("KO-IV", "w"):
#    pickle.dump(valuesf, f)