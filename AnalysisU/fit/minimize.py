import numpy as np
import multiprocessing
from L import L
from admin import make_ps

Const=([2.42113876e-01, 1.67631209e-01, 1.58076323e-01, 2.16066806e-01,
 2.00606154e-01, 2.25645993e-01, 4.07734489e+01, 4.13619837e+01,
 4.05362765e+01, 4.04138555e+01, 4.07258581e+01, 4.11255294e+01,
 5.46874571e-01, 8.21390422e-01, 4.26925194e-01, 7.19220196e-01,
 5.05166708e-01, 7.48014047e-01, 3.71239620e-01, 2.09395868e-01,
 1.79299737e+01, 1.06977242e+02, 1.67719833e+00, 4.77128877e-01,
 4.24762658e-01, 3.41243978e-01, 2.82956642e-01, 5.06594842e-02,
 9.21813118e+00, 2.78242776e+01])

def minimize(pmt, q, ID):
    stop=0
    ps=make_ps()

    ps[:,18:20]=Const[18:20]

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
        while len(ls)<1550:
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
