# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 22:01:37 2018

@author: Flo
"""

import numpy as np
import matplotlib.pyplot as plt

"""
solves uxx = f on (a,b)
with u(a) = ua and u(b) = ub

utilizing straight forward intgration, ie euler method, ie theta = 0, ie boring and stupid
note the sign reversal
"""


f = lambda x : - np.sin(np.pi * x)
a = 0
b = 1
ua = 0
ub = 0

N = 100

def forwardEuler(f, a, b, ua, ub, N):
        
    h = (a-b)/N
    x = np.linspace(a, b, num = N+1)
    
    vxx = f(x)             # = uxx
    vx = np.cumsum(vxx)*h   # = ux + c1
    v = np.cumsum(vx)*h     # = u + c1*x + c2

    
    """
    u(a) = ua = v(a) - c1*a - c2
    u(b) = ub = v(b) - c1*b - c2
    
    
    |a 1|   |c1|   |v(a) - ua|
    |b 1| * |c2| = |v(b) - ub|
    \:=A/
    
    A^-1 = 1/(a-b) * |1 -1|
                     |-b a|
    
    """
    
    c1 = 1/(a-b) * (   (v[0] - ua) -   (v[N] - ub))
    c2 = 1/(a-b) * (-b*(v[0] - ua) + a*(v[N] - ub))
    
    u = v - c1*x - c2
    
    return u, x
    
if __name__ == "__main__":
    
    u, x = forwardEuler(f, a, b, ua, ub, N)
    
    plt.plot(x, u)
    plt.plot(x,1/np.pi**2 * np.sin(np.pi * x))
    plt.legend(["numerical", "exact"])
