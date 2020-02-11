import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import scipy as sci
import scipy.special as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors

good_chns=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
r=22/40
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

C=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'rosybrown', 'indigo', 'pink', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'rosybrown', 'indigo', 'pink']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for J in range(len(pmt_mid)):
   for J in range(len(pmt_mid)):
    c=J
    if J<10:
        al=1
    else:
        al=1
    ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],
        pmt_up[J,1], pmt_up[J,2], color=C[c], linewidth=5, alpha=al)
    ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],
        pmt_dn[J,1], pmt_dn[J,2], color=C[c], linewidth=5, alpha=al)
    ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],
        pmt_up[J,1], pmt_up[J,2], color=C[c], linewidth=5, alpha=al)
    ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],
        pmt_dn[J,1], pmt_dn[J,2], color=C[c], linewidth=5, alpha=al)

    ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],
        pmt_r[J,1], pmt_r[J,2], color=C[c], linewidth=5, alpha=al)
    ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],
        pmt_l[J,1], pmt_l[J,2], color=C[c], linewidth=5, alpha=al)
    ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],
        pmt_r[J,1], pmt_r[J,2], color=C[c], linewidth=5, alpha=al)
    ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],
        pmt_l[J,1], pmt_l[J,2], color=C[c], linewidth=5, alpha=al)

x_int=0.5
y_int=0
z_int=0
while len(pmt)<500:
    print(len(pmt))
    costheta=np.random.uniform(-1,1)
    phi=np.random.uniform(0,2*np.pi)
    v=np.array([x_int+np.sin(np.arccos(costheta))*np.cos(phi), y_int+np.sin(np.arccos(costheta))*np.sin(phi), z_int+costheta])
    alpha=np.zeros(len(pmt_mid))+1e6
    for j in range(len(pmt_mid)):
        if np.sum(v*pmt_mid[j])>0:
            alpha[j]=1/(np.sum(v*pmt_mid[j]))
    J=np.argmin(alpha)
    Alpha=np.amin(alpha)

    if (np.sum((pmt_mid[J]-Alpha*v)*pmt_r[J])<r**2 and np.sum((pmt_mid[J]-Alpha*v)*pmt_r[J])>-r**2 and
            np.sum((pmt_mid[J]-Alpha*v)*pmt_up[J])<r**2 and np.sum((pmt_mid[J]-Alpha*v)*pmt_up[J])>-r**2 and J in good_chns):
            # ax.scatter(Alpha*v[0],Alpha*v[1],Alpha*v[2], marker='o', color='r')
            ax.quiver(0, 0, 0, 0.5*v[0],  0.5*v[1], 0.5*v[2], color='r')
            pmt=np.append(pmt,J)
            for j in range(len(pmt)-1):
                ang=np.append(ang, 180*np.arccos(np.sum(pmt_mid[pmt[j]]*pmt_mid[J]))/np.pi)

#pmt=np.repeat(np.arange(20),30)
#pmt=np.array([])
fig = plt.figure()
ax = fig.add_subplot(111)
bins=np.array([-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])+0.5
h,bins,batch = ax.hist(pmt, bins=bins)
ax.set_xlabel('PMT')
ax.set_ylabel('Nsumber of hits')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('$|F_l^m|$', fontsize=20)
ax.set_xlabel('l')
ax.set_ylabel('m')
PHI, THETA = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
Fml=np.zeros((5,3), dtype=np.complex_)
#Fml=np.zeros((3,2), dtype=np.complex_)
for l in range(len(Fml[0,:])):
    for m in np.arange(-l,l+1):
        for p in range(20):
            if pmt_mid[p,0]>=0:
                phi=np.arctan(pmt_mid[p,1]/pmt_mid[p,0])
            else:
                phi=np.pi+np.arctan(pmt_mid[p,1]/pmt_mid[p,0])
            theta=np.arccos(pmt_mid[p,2]/np.sqrt(np.sum(pmt_mid[p,:]**2)))
            # if h[p]>0:
            #     print(180*theta/np.pi, 180*phi/np.pi)
            Fml[l+m,l]+=h[p]*np.conj(sp.sph_harm(m, l, phi, theta))
        ax.scatter(l,m,np.abs(Fml[l+m,l]), c='k')

########## Find dipole axis#####################
# Yml=0*sp.sph_harm(0, 0, PHI, THETA)
# if np.any(np.abs(Fml[:,1])>1e-10):
#     l=1
# elif np.any(np.abs(Fml[:,2])>1e-10):
#     l=2
#     print('No dipol')
# for m in np.arange(-l,l+1):
#     Yml+=Fml[l+m,l]*sp.sph_harm(m, l, PHI, THETA)
# R=Yml.real
# R=R/R.max()
# R_max=np.amax(R)
# R_min=np.amin(1+R)
# PHI_max=PHI[int(np.floor(np.argmax(R)/len(THETA[0,:]))),0]
# THETA_max=THETA[0,int(np.argmax(R)%len(THETA[0,:]))]
# PHI_min=PHI[int(np.floor(np.argmin(R)/len(THETA[0,:]))),0]
# THETA_min=THETA[0,int(np.argmin(R)%len(THETA[0,:]))]
# X_max = R_max * np.sin(THETA_max) * np.cos(PHI_max)
# Y_max = R_max * np.sin(THETA_max) * np.sin(PHI_max)
# Z_max = R_max * np.cos(THETA_max)
# X_min = R_min * np.sin(THETA_min) * np.cos(PHI_min)
# Y_min = R_min * np.sin(THETA_min) * np.sin(PHI_min)
# Z_min = R_min * np.cos(THETA_min)

Yml=0*sp.sph_harm(0, 0, PHI, THETA)
for l in range(0,len(Fml[0,:])):
    for m in np.arange(-l,l+1):
        if 1==1:
            Yml+=Fml[l+m,l]*sp.sph_harm(m, l, PHI, THETA)
Norm=np.abs(Yml)
R=Yml.real
I=Yml.imag
R_maxi=R.max()
Norm_maxi=Norm.max()
R=R/R.max()
Norm=Norm/Norm.max()
R_max=np.amax(R)
R_min=np.amin(1+R)
PHI_max=PHI[int(np.floor(np.argmax(R)/len(THETA[0,:]))),0]
THETA_max=THETA[0,int(np.argmax(R)%len(THETA[0,:]))]
PHI_min=PHI[int(np.floor(np.argmin(R)/len(THETA[0,:]))),0]
THETA_min=THETA[0,int(np.argmin(R)%len(THETA[0,:]))]
X_max = R_max * np.sin(THETA_max) * np.cos(PHI_max)
Y_max = R_max * np.sin(THETA_max) * np.sin(PHI_max)
Z_max = R_max * np.cos(THETA_max)
X_min = R_min * np.sin(THETA_min) * np.cos(PHI_min)
Y_min = R_min * np.sin(THETA_min) * np.sin(PHI_min)
Z_min = R_min * np.cos(THETA_min)


s=0.25
r=0.25
X = (s*R+r) * np.sin(THETA) * np.cos(PHI)
Y = (s*R+r) * np.sin(THETA) * np.sin(PHI)
Z = (s*R+r) * np.cos(THETA)


norm = colors.Normalize()
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(14,10))
#ax.quiver(X_min-X_max, Y_min-Y_max, Z_min-Z_max, 2*X_max-X_min, 2*Y_max-Y_min, 2*Z_max-Z_min, color='k')
for p in pmt:
    ax.scatter(pmt_mid[p,0], pmt_mid[p,1], pmt_mid[p,2], c='r')
for J in range(len(pmt_mid)):
    if J in good_chns:
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')

        ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],
            pmt_up[J,1], pmt_up[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],
            pmt_dn[J,1], pmt_dn[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],
            pmt_up[J,1], pmt_up[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],
            pmt_dn[J,1], pmt_dn[J,2], color='g')

        ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],
            pmt_r[J,1], pmt_r[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],
            pmt_l[J,1], pmt_l[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],
            pmt_r[J,1], pmt_r[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],
            pmt_l[J,1], pmt_l[J,2], color='g')
    else:
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

        ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],
            pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],
            pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],
            pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],
            pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

        ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],
            pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],
            pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],
            pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],
            pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
m = cm.ScalarMappable(cmap=cm.jet)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(norm(R)))
ax.set_title('0.5+0.5*real$(Y^m_l) $'+'(max={})'.format(R_maxi), fontsize=20)
#ax.legend(fontsize=15)
m.set_array(R)
fig.colorbar(m, shrink=0.8);

#
# X = (s*I+r) * np.sin(THETA) * np.cos(PHI)
# Y = (s*I+r) * np.sin(THETA) * np.sin(PHI)
# Z = (s*I+r) * np.cos(THETA)
# norm = colors.Normalize()
# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(14,10))
# for p in pmt:
#     ax.scatter(pmt_mid[p,0], pmt_mid[p,1], pmt_mid[p,2], c='r')
# for J in range(len(pmt_mid)):
#     if J in good_chns:
#         ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
#         ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
#         ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
#         ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')
#
#         ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],
#             pmt_up[J,1], pmt_up[J,2], color='g')
#         ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],
#             pmt_dn[J,1], pmt_dn[J,2], color='g')
#         ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],
#             pmt_up[J,1], pmt_up[J,2], color='g')
#         ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],
#             pmt_dn[J,1], pmt_dn[J,2], color='g')
#
#         ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],
#             pmt_r[J,1], pmt_r[J,2], color='g')
#         ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],
#             pmt_l[J,1], pmt_l[J,2], color='g')
#         ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],
#             pmt_r[J,1], pmt_r[J,2], color='g')
#         ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],
#             pmt_l[J,1], pmt_l[J,2], color='g')
#     else:
#         ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
#         ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
#         ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
#         ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)
#
#         ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],
#             pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
#         ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],
#             pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)
#         ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],
#             pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
#         ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],
#             pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)
#
#         ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],
#             pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
#         ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],
#             pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
#         ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],
#             pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
#         ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],
#             pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
# m = cm.ScalarMappable(cmap=cm.jet)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(norm(R)))
# ax.set_title('0.5+0.5*Imag$(Y^m_l)$', fontsize=20)
# #ax.legend(fontsize=15)
# m.set_array(R)
# fig.colorbar(m, shrink=0.8);






X = (s*Norm+r) * np.sin(THETA) * np.cos(PHI)
Y = (s*Norm+r) * np.sin(THETA) * np.sin(PHI)
Z = (s*Norm+r) * np.cos(THETA)


norm = colors.Normalize()
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(14,10))
#ax.quiver(X_min-X_max, Y_min-Y_max, Z_min-Z_max, 2*X_max-X_min, 2*Y_max-Y_min, 2*Z_max-Z_min, color='k')
for p in pmt:
    ax.scatter(pmt_mid[p,0], pmt_mid[p,1], pmt_mid[p,2], c='r')
for J in range(len(pmt_mid)):
    if J in good_chns:
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='g')
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='g')
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='g')
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='g')

        ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],
            pmt_up[J,1], pmt_up[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],
            pmt_dn[J,1], pmt_dn[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],
            pmt_up[J,1], pmt_up[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],
            pmt_dn[J,1], pmt_dn[J,2], color='g')

        ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],
            pmt_r[J,1], pmt_r[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],
            pmt_l[J,1], pmt_l[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],
            pmt_r[J,1], pmt_r[J,2], color='g')
        ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],
            pmt_l[J,1], pmt_l[J,2], color='g')
    else:
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_r[J,0],  pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_l[J,0],  pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_up[J,0],  pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0], pmt_mid[J,1], pmt_mid[J,2],pmt_dn[J,0],  pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

        ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_up[J,0],
            pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_r[J,0], pmt_mid[J,1]+pmt_r[J,1], pmt_mid[J,2]+pmt_r[J,2],pmt_dn[J,0],
            pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_up[J,0],
            pmt_up[J,1], pmt_up[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_l[J,0], pmt_mid[J,1]+pmt_l[J,1], pmt_mid[J,2]+pmt_l[J,2],pmt_dn[J,0],
            pmt_dn[J,1], pmt_dn[J,2], color='k', alpha=0.2)

        ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_r[J,0],
            pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_up[J,0], pmt_mid[J,1]+pmt_up[J,1], pmt_mid[J,2]+pmt_up[J,2],pmt_l[J,0],
            pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_r[J,0],
            pmt_r[J,1], pmt_r[J,2], color='k', alpha=0.2)
        ax.quiver(pmt_mid[J,0]+pmt_dn[J,0], pmt_mid[J,1]+pmt_dn[J,1], pmt_mid[J,2]+pmt_dn[J,2],pmt_l[J,0],
            pmt_l[J,1], pmt_l[J,2], color='k', alpha=0.2)
m = cm.ScalarMappable(cmap=cm.jet)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(norm(Norm)))
ax.set_title('0.5+0.5*Norm$(Y^m_l)$'+'(max={})'.format(Norm_maxi), fontsize=20)
#ax.legend(fontsize=15)
m.set_array(Norm)
fig.colorbar(m, shrink=0.8);


plt.show()
