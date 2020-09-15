import numpy as np
import matplotlib.pyplot as plt
import sys

r=21/40
pmt_mid=np.array([[1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), 1/np.sqrt(2)],
                [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), 1/np.sqrt(2)],
                [0,1,0],
                [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                [-1,0,0],
                [-1/np.sqrt(2), -1/np.sqrt(2),0],
                [1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), -1/np.sqrt(2)],
                [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), -1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), 1/np.sqrt(2)],
                [1/np.sqrt(2), 0,1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), 1/np.sqrt(2)],
                [0, -1, 0],
                [1/np.sqrt(2), -1/np.sqrt(2), 0],
                [1, 0, 0],
                [1/np.sqrt(2), 1/np.sqrt(2), 0],
                [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), -1/np.sqrt(2)],
                [1/np.sqrt(2), 0, -1/np.sqrt(2)],
                [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), -1/np.sqrt(2)],
                ])


pmt_r=np.zeros((20,3))
pmt_up=np.zeros((20,3))

for i, pmt in enumerate((range(20))):
    a=pmt_mid[pmt][0]
    b=pmt_mid[pmt][1]
    pmt_r[i]=[-0.5*r*b/np.sqrt(a**2+b**2), 0.5*r*a/np.sqrt(a**2+b**2), 0]
    pmt_up[i]=np.cross(pmt_mid[pmt], pmt_r[i])

rt=pmt_r
up=pmt_up
mid=pmt_mid

N=1000

rat=[]
for j in range(1000):
    Ind=[]
    costheta=np.random.uniform(-1,1, N)
    phi=np.random.uniform(0,2*np.pi, N)
    us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
    #us=np.array([0.74920942, -0.11109017,  0.65295039]).reshape(3,1)
    vs=np.repeat(np.array([0,0,0]), N).reshape(3, N)
    hits=np.zeros(len(us[0]))-1
    for i in range(len(mid)):
        a=(1-np.sum(vs.T*mid[i], axis=1))/np.sum(us.T*mid[i], axis=1) # N length
        r=(vs+a*us).T-mid[i] # (N,3)
        ind=np.nonzero(np.logical_and(a>0, np.logical_and(np.abs(np.sum(r*rt[i], axis=1))<np.sum(rt[i]**2), np.abs(np.sum(r*up[i], axis=1))<np.sum(up[i]**2))))[0]
        #print(i, ind)
        if np.any(np.isin(ind, np.array(Ind))):
            ind0=ind[np.isin(ind, np.array(Ind))]
            print('Fuck')
            print(us[:,ind0[0]])
            print(np.nonzero(np.array(Ind)==ind0[0])[0])
            print(i, np.unique(hits[Ind[np.nonzero(np.array(Ind)==ind0[0])[0][0]]]))
            sys.exit()
        Ind.extend(list(ind))
        hits[ind]=i
    rat.append(len(hits[hits>=0])/N)

plt.figure()
plt.hist(rat, bins=100)
plt.show()
