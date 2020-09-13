import numpy as np
import multiprocessing
from L import L


def rec_to_p(rec):
    p=np.array([])
    for name in rec.dtype.names:
        p=np.append(p, np.array(rec[name][0]))
    return p

def minimize(rec, param, ind, param_name):
    ls=[]
    PS=np.zeros((100, 6))
    stop=0
    count=0
    ps=make_ps(rec_to_p(rec))
    ps[:,ind]=param
    l=np.zeros(np.shape(ps)[0])
    for i in range(len(l)):
        print('Init minimizing', i, 'out of', len(l))
        l[i]=L(ps[i], param, ind, param_name, PS, ls)
    while len(ls)<np.shape(PS)[0]:
        print(param_name,'=', param, count)
        count+=1
        h=0.5
        a=1
        g=1
        s=0.5
        ind=np.argsort(l)
        m=np.mean(ps[ind[:-1]], axis=0)

        r=m+a*(m-ps[ind[-1]])
        lr=L(r, param, ind, param_name, PS, ls)
        if l[ind[0]]<lr and lr<l[ind[-2]]:
            ps[ind[-1]]=r
            l[ind[-1]]=lr
        elif lr<l[ind[0]]:
            e=m+g*(r-m)
            le=L(e, param, ind, param_name, PS, ls)
            if le<lr:
                ps[ind[-1]]=e
                l[ind[-1]]=le
            else:
                ps[ind[-1]]=r
                l[ind[-1]]=lr
        else:
            c=m+h*(ps[ind[-1]]-m)
            lc=L(c, param, ind, param_name, PS, ls)
            if lc<l[ind[-1]]:
                ps[ind[-1]]=c
                l[ind[-1]]=lc
            else:
                for i in ind[1:]:
                    ps[i]=ps[ind[0]]+s*(ps[i]-ps[ind[0]])
                    l[i]=L(ps[i], param, ind, param_name, PS, ls)

def make_ps(p):
    ps=np.zeros((len(p), len(p)))
    ps[:,:6]=np.random.uniform(0.1, 0.3, size=len(np.ravel(ps[:,:6]))).reshape(np.shape(ps[:,:6]))
    return ps
