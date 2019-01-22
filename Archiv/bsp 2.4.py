# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:30:15 2018

@author: Flo
"""

import sympy as sp
sp.init_printing(use_unicode = True, forecolor = "White")

S, K, t, T, sigma, r, x = sp.symbols("S K t T \sigma r x")
#d1, d2 = sp.symbols("d_1 d_2", cls = sp.Function)
#phi, C, P = sp.symbols("\Phi C P", cls = sp.Function)

d1 = (sp.ln(S/K) + (r + sigma**2/2)*(T-t)) / (sigma * sp.sqrt(T-t))
d2 = (sp.ln(S/K) + (r - sigma**2/2)*(T-t)) / (sigma * sp.sqrt(T-t))

phi = sp.stats.cdf(sp.stats.Normal("x", 0, 1))

C = S * phi(d1) - K*sp.exp(-r*(T-t)) * phi(d2)
P = K * sp.exp(r * (T-t)) * phi(-d2) - S * phi(-d1)

deltaC = C.diff(S)
gammaC = C.diff(S, S)
kappaC = C.diff(sigma)
thetaC = C.diff(t)
rhoC = C.diff(r)

deltaP = P.diff(S)
gammaP = P.diff(S, S)
kappaP = P.diff(sigma)
thetaP = P.diff(t)
rhoP = P.diff(r)

