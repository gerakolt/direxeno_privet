import numpy as np

def Sim(T, St, bins):
    N=10000
    ts=np.zeros((len(T), N))
    for i in range(len(T)):
        ts[i]=np.random.normal(T[i], St[i], N)

    H=np.zeros((15,len(bins)-1))
    k=0
    for i in range(len(T)-1):
        for j in range(i+1, len(T)):
            H[k]=np.histogram(ts[j]-ts[i], bins=bins)[0]
            k+=1
    return (H.T/np.amax(H, axis=1)).T
