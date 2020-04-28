import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import poisson, binom
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.special import erf as erf
import sys
import random
from datetime import datetime
random.seed(datetime.now())


def make_P(Spe, p01):
    n=100
    P=np.zeros((n,n))
    P[0,1]=p01
    P[1,1]=0.5*(1+erf(0.5/(np.sqrt(2)*Spe)))-p01
    for i in np.arange(2, n):
        P[i,1]=0.5*(erf((i-0.5)/(np.sqrt(2)*Spe))-erf((i-1.5)/(np.sqrt(2)*Spe)))

    for i in range(n):
        for j in range(2,n):
            P[i,j]=np.sum(P[:i+1,1]*np.flip(P[:i+1,j-1], axis=0))
    # P[:,0]=1-np.sum(P[:,1:], axis=1)
    F=np.zeros((n,n))
    F[0,0]=1-np.sum(P[0,:])
    F[0,1:]=P[0,1:]
    F[1:,0]=((1-F[0,0])/(2-F[0,0]))**(np.arange(1, len(F[:,0])))
    for i in range(1,n):
        for j in range(1,n):
            F[i,j]=np.sum(F[:i+1,0]*np.flip(P[:i+1,j], axis=0))

    return F



# Spe=np.random.uniform(0,1,50)
# p01=np.random.uniform(0,1,50)
# fig, ((ax1, ax2), (ax3, ax4))=plt.subplots(2,2)
# for i, spe in enumerate(Spe):
#     for j, p in enumerate(p01):
#         print(i,j)
#         P=make_P(spe, p)
#         if np.shape(P)==(3,):
#             if P[0]>0.5:
#                 ax1.scatter(spe, P[2], color='k')
#                 ax2.scatter(p, P[2], color='k')
#                 ax3.scatter(p, P[1], color='k')
#                 ax4.scatter(spe, P[1], color='k')
#                 ax3.set_xlabel('P01')
#                 ax3.set_ylabel('P01+P11')
#             elif P[0]<0:
#                 ax1.scatter(spe, P[2], color='r')
#                 ax2.scatter(p, P[2], color='r')
#                 ax3.scatter(p,P[1], color='r')
#                 ax4.scatter(spe, P[1], color='r')
#                 ax4.set_xlabel('Spe')
#                 ax4.set_ylabel('P01+P11')
# plt.show()


P=make_P(0.5,0.2)
plt.plot(np.sum(P, axis=0), '.', label='axis 0')
plt.plot(np.sum(P, axis=1), '.', label='axis 1')
plt.legend()
plt.yscale('log')
plt.show()
