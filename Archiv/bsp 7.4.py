# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 03:04:43 2018

@author: Flo
"""

import numpy as np
import matplotlib.pyplot as plt
from bspSiebenPunktVierAberInSchlecht import forwardEuler

"""
solves uxx = f on (a,b)
with u(a) = ua and u(b) = ub

using fourth order compact scheme from lecture notes p.103 or exercise 7.2
beware the sign of f
"""


f = lambda x : - np.sin(np.pi * x)
a = 0
b = 1
ua = 0
ub = 0

N = 10
h = (a-b)/N
x = np.linspace(a, b, num = N+1)

"""
lecture notes say:
1/h^2 * (w_i+1 - 2w_i + w_i-1) = 1/12 * (f(x_i+1) + 10*f(x_i) + f(x_i-1))

in matrix-language: .l and .r denotes left and right shifts
1/h^2 * tridiag(1, -2, 1) * w = 1/12 * (f(x.l) + 10*f(x) + f(x.r))
for inner elements of w
w_0 = ua, w_N = ub

A * x = B
"""

A = 1/h**2 * (np.diag(np.ones(N), k=-1) + np.diag(-2*np.ones(N+1), k=0) + np.diag(np.ones(N), k=1))
A[0,0] = 1
A[0,1] = 0
A[N,N] = 1
A[N,N-1] = 0

B = np.array([ua] + list(1/12 * (f(x[:-2]) + 10 * f(x[1:-1]) + f(x[2:]))) + [ub])

w = np.linalg.solve(A,B)

u, x = forwardEuler(f, a, b, ua, ub, N)

plt.plot(x, 1/np.pi**2 * np.sin(np.pi*x))
plt.plot(x, u)
plt.plot(x, w)

plt.legend(["analytic", "bad numeric", "good numeric"])