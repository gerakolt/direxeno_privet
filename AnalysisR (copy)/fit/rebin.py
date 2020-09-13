import numpy as np
# def rebin_spectra(S):
#     q=np.shape(S)[-1]
#     n=2
#     N=int(100/n)
#     s=np.zeros((n, q))
#     for i in range(q):
#         for j in range(n):
#             s[j,i]=np.sum(S[N*j:N*(j+1), i])
#     return np.arange(n)*N, s



def rebin_spectrum(S):
    n=20
    s=np.zeros(len(S)//n)
    bins=[0]
    for i in range(len(S)//n):
        s[i]=np.sum(S[n*i:n*(i+1)])
        bins.append(n*(i+1))
    return np.array(bins), s
