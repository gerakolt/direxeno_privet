import numpy as np
def rebin_spectra(S):
    q=np.shape(S)[-1]
    n=5
    N=int(100/n)
    s=np.zeros((n, q))
    for i in range(q):
        for j in range(n):
            s[j,i]=np.sum(S[N*j:N*(j+1), i])
    return np.arange(n)*N, s


# def rebin_spectra(S):
#     q=np.shape(S)[-1]
#     n=70
#     s=np.zeros((n, q))
#     for i in range(q):
#         for j in range(n):
#             s[j,i]=np.sum(S[5*j:5*(j+1), i])
#     return np.arange(n)*5, s
