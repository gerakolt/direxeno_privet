import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

good_chns=[0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,19]
r=0.25
pmt=np.array([]).astype(int)
ang=np.array([])
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

pmt_r=np.zeros((len(pmt_mid),3))
pmt_l=np.zeros((len(pmt_mid),3))
pmt_up=np.zeros((len(pmt_mid),3))
pmt_dn=np.zeros((len(pmt_mid),3))

for i in range(len(pmt_mid)):
    a=pmt_mid[i][0]
    b=pmt_mid[i][1]
    pmt_r[i]=[-r*b/np.sqrt(a**2+b**2), r*a/np.sqrt(a**2+b**2), 0]
    pmt_l[i]=[r*b/np.sqrt(a**2+b**2),-r*a/np.sqrt(a**2+b**2), 0]
    pmt_up[i]=np.cross(pmt_mid[i], pmt_r[i])
    pmt_dn[i]=np.cross(pmt_mid[i], pmt_l[i])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)

for J in range(len(pmt_mid)):
    if J!=3 and J!=4:
        if J in good_chns:
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')

            ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')

            ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
            ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
        else:
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

            ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

            ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
            ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)

#for i in range(1000):
Costheta=np.random.uniform(-1,1)
Phi=np.random.uniform(0,2*np.pi)
while len(pmt)<315:
    print(len(pmt))
    costheta=np.random.normal(loc=Costheta, scale=1, size=1)
    phi=np.random.normal(loc=Phi, scale=3, size=1)
    while not (costheta>-1 and costheta<1):
        costheta=np.random.normal(loc=Costheta, scale=0.5, size=1)
    while not (phi>0 and phi<2*np.pi):
        phi=np.random.normal(loc=Phi, scale=3, size=1)
    v=np.array([np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi), costheta])
    alpha=np.zeros(len(pmt_mid))+1e6
    for j in range(len(pmt_mid)):
        if np.sum(v*pmt_mid[j])>0:
            alpha[j]=1/(np.sum(v*pmt_mid[j]))
    J=np.argmin(alpha)
    Alpha=np.amin(alpha)
    # J=15
    # alpha=alpha[J]
    if (np.sum((pmt_mid[J]-Alpha*v)*pmt_r[J])<r**2 and np.sum((pmt_mid[J]-Alpha*v)*pmt_r[J])>-r**2 and
            np.sum((pmt_mid[J]-Alpha*v)*pmt_up[J])<r**2 and np.sum((pmt_mid[J]-Alpha*v)*pmt_up[J])>-r**2 and J in good_chns):
            ax.scatter(Alpha*v[0],Alpha*v[1],Alpha*v[2], marker='o', color='r')
            ax.quiver(0, 0, 0, 0.5*v[0],  0.5*v[1], 0.5*v[2], color='r')
            pmt=np.append(pmt,J)
            for j in range(len(pmt-1)):
                ang=np.append(ang, 180*np.arccos(np.sum(pmt_mid[pmt[j]]*pmt_mid[J]))/np.pi)




plt.show()
h,bins,p = plt.hist(ang, label='Simulated isotropic scintillation', bins=180, range=[-0.5,180.5])
plt.xlabel('Angle [deg]', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=25)
plt.legend(loc='best', fontsize=25)
plt.show()
plt.plot(0.5*(bins[:-1]+bins[1:]),h, 'o')
plt.errorbar(0.5*(bins[:-1]+bins[1:]), h, yerr=np.sqrt(h), ls='')
plt.axhline(y=np.mean(h), xmin=0, xmax=1)
plt.xlabel('PMT', fontsize='25')
plt.legend(loc='best', fontsize=25)
plt.show()
