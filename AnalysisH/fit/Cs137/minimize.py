import numpy as np


def minimize(L, ps):
    l=np.zeros(np.shape(ps)[0])
    for i in range(len(l)):
        l[i]=L(ps[i])
    stop=0
    while not stop:
        h=0.5
        a=1
        g=1
        s=0.5
        ind=np.argsort(l)
        m=np.mean(ps[ind[:-1]], axis=0)

        r=m+a*(m-ps[ind[-1]])
        lr=L(r)
        if l[ind[0]]<lr and lr<l[ind[-2]]:
            ps[ind[-1]]=r
            l[ind[-1]]=lr
        elif lr<l[ind[0]]:
            e=m+g*(r-m)
            le=L(e)
            if le<lr:
                ps[ind[-1]]=e
                l[ind[-1]]=le
            else:
                ps[ind[-1]]=r
                l[ind[-1]]=lr
        else:
            c=m+h*(ps[ind[-1]]-m)
            lc=L(c)
            if lc<l[ind[-1]]:
                ps[ind[-1]]=c
                l[ind[-1]]=lc
            else:
                for i in ind[1:]:
                    ps[i]=ps[ind[0]]+s*(ps[i]-ps[ind[0]])
                    l[i]=L(ps[i])


def make_ps(p):
    ps=np.zeros((len(p)+1, len(p)))
    ps[0]=p
    ps[1:,:6]=np.random.uniform(0.01, 1, size=len(np.ravel(ps[1:,:6]))).reshape(np.shape(ps[1:,:6]))
    ps[1:,6:12]=np.random.uniform(40, 50, size=len(np.ravel(ps[1:,:6]))).reshape(np.shape(ps[1:,:6]))
    ps[1:,12:18]=np.random.uniform(0.1, 3, size=len(np.ravel(ps[1:,:6]))).reshape(np.shape(ps[1:,:6]))
    ps[1:, 18]=np.random.uniform(0.001, 1, size=len(ps[1:, 18]))
    ps[1:, 19]=np.random.uniform(0.5, 15, size=len(ps[1:, 18]))
    ps[1:, 20]=np.random.uniform(15, 50, size=len(ps[1:, 18]))
    ps[1:, 21]=np.random.uniform(0.001, 1, size=len(ps[1:, 18]))
    ps[1:, 22]=np.random.uniform(0.001,1, size=len(ps[1:, 18]))
    ps[1:, 23]=np.random.uniform(0.001,1, size=len(ps[1:, 18]))
    return ps
