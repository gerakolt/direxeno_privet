import numpy as np

def Sim(Ma, Sa, q, th, Abins, DCSpec, DCAreas):
    N_events=10000
    S=np.zeros(N_events)
    s=np.zeros(N_events)
    Areas=np.zeros(len(Abins)-1)
    for i in range(N_events):
        As=[]
        N=np.random.poisson(q)
        DCN=np.random.choice(np.arange(len(DCSpec)), size=1, p=DCSpec/np.sum(DCSpec)).astype(int)
        areas=np.random.normal(Ma, Sa, size=N)
        dcareas=np.random.choice(0.5*(Abins[1:]+Abins[:-1]),size=DCN, p=DCAreas/np.sum(DCAreas))
        area=np.sum(areas)+np.sum(dcareas)
        n=0
        while area>th:
            As.append(area)
            if area<1:
                n+=1
                area=0
            else:
                area-=1
                n+=1
        S[i]=n
        Areas+=np.histogram(As, bins=Abins)[0]
    H=np.histogram(S, bins=np.arange(len(DCSpec)+1))[0]
    return H/N_events, Areas/np.sum(Areas)
