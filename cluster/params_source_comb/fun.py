from scipy.optimize import curve_fit
import numpy as np

def fit_ls(ls):
    def exp(x, a,b):
        return a*np.exp(-x/b)
    def lin(x, a,b):
        return -a*x+b
    mu=np.zeros(len(ls))-1
    a=np.zeros(len(ls))-1
    v=np.zeros(len(ls))-1
    x=np.arange(150)
    for i in range(75, len(ls)-75):
        v[i]=np.var(ls[i-75:i+75])
        try:
            p, cov=curve_fit(exp, x, ls[i-75:i+75])
            mu[i]=p[1]
        except:
            continue
        try:
            p, cov=curve_fit(lin, x, ls[i-75:i+75])
            a[i]=p[0]
        except:
            continue
    return v, mu, a
