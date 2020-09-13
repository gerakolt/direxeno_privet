import numpy as np
def rebin_spectra(S):
    q=np.shape(S)[-1]
    n=20
    s=np.zeros((n, q))
    for i in range(q):
        for j in range(n):
            s[j,i]=np.sum(S[5*j:5*(j+1), i])
    return np.arange(20)*5, s
