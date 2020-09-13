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


def rebin_spectra(S):
    q=np.shape(S)[-1]
    s=np.zeros((2, q))
    for i in range(q):
        h=S[:,i]
        s[0,i]=np.sum(h[:50])
        s[1,i]=np.sum(h[50:])

    return np.array([0, 50, len(h)]), s
