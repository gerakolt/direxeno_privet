import numpy as np
import multiprocessing
from L import L


def minimize(pmt, q, ID):
    stop=0
    ps=make_ps()
    PS=np.zeros((1000, np.shape(ps)[1]))
    ls=[]
    while not stop:
        stop=1
        count=0
        if pmt>=0:
            ps[:,pmt]=q
        l=np.zeros(np.shape(ps)[0])
        for i in range(len(l)):
            l[i]=L(ps[i], pmt, q, PS, ls, ID)
        while len(ls)<550:
            count+=1
            h=0.5
            a=1
            g=1
            s=0.5
            ind=np.argsort(l)
            m=np.mean(ps[ind[:-1]], axis=0)

            r=m+a*(m-ps[ind[-1]])
            lr=L(r, pmt, q, PS, ls, ID)
            if l[ind[0]]<lr and lr<l[ind[-2]]:
                ps[ind[-1]]=r
                l[ind[-1]]=lr
            elif lr<l[ind[0]]:
                e=m+g*(r-m)
                le=L(e, pmt, q, PS, ls, ID)
                if le<lr:
                    ps[ind[-1]]=e
                    l[ind[-1]]=le
                else:
                    ps[ind[-1]]=r
                    l[ind[-1]]=lr
            else:
                c=m+h*(ps[ind[-1]]-m)
                lc=L(c, pmt, q, PS, ls, ID)
                if lc<l[ind[-1]]:
                    ps[ind[-1]]=c
                    l[ind[-1]]=lc
                else:
                    for i in ind[1:]:
                        ps[i]=ps[ind[0]]+s*(ps[i]-ps[ind[0]])
                        l[i]=L(ps[i], pmt, q, PS, ls, ID)


def make_ps():
    n=6
    ps=np.zeros((n+1, n))
    ps[:,:6]=np.random.uniform(0.1, 0.3, size=len(np.ravel(ps[:,:6]))).reshape(np.shape(ps[:,:6]))
    return ps
