from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys

dt=10**(-16)
# dt=10**(-15)
fc=2.3*10**(-28) #N=Kg m^3/s^2
me=9.1*10**(-31) #kg
mn=130*1.672*10**(-27) #kg
R=10**(-8)
V=R*10**12
print('Ve sould be,', fc*dt/(me*R**3),'R')
print('If you want 10^-10 m steps, dt=', np.sqrt(10**(-30)*me/fc))

def make_f(e,n):
    F_ee_x=np.zeros((len(e[:,0]), len(n[:,0])))
    F_ee_y=np.zeros((len(e[:,0]), len(n[:,0])))
    F_ee_z=np.zeros((len(e[:,0]), len(n[:,0])))
    F_nn_x=np.zeros((len(e[:,0]), len(n[:,0])))
    F_nn_y=np.zeros((len(e[:,0]), len(n[:,0])))
    F_nn_z=np.zeros((len(e[:,0]), len(n[:,0])))
    F_en_x=np.zeros((len(e[:,0]), len(n[:,0])))
    F_en_y=np.zeros((len(e[:,0]), len(n[:,0])))
    F_en_z=np.zeros((len(e[:,0]), len(n[:,0])))
    for i in range(len(F_ee_x)-1):
        for j in range(i+1, len(F_ee_x)):
            r=np.sqrt((e[i,0]-e[j,0])**2+(e[i,1]-e[j,1])**2+(e[i,2]-e[j,2])**2)
            if r==0:
                continue
            else:
                F_ee_x[i,j]=fc*np.abs(e[i,0]-e[j,0])/(r)**3
                F_ee_x[j,i]=fc*np.abs(e[i,0]-e[j,0])/(r)**3
                F_ee_y[i,j]=fc*np.abs(e[i,1]-e[j,1])/(r)**3
                F_ee_y[j,i]=fc*np.abs(e[i,1]-e[j,1])/(r)**3
                F_ee_z[i,j]=fc*np.abs(e[i,2]-e[j,2])/(r)**3
                F_ee_z[j,i]=fc*np.abs(e[i,2]-e[j,2])/(r)**3

            r=np.sqrt((e[i,0]-e[j,0])**2+(e[i,1]-e[j,1])**2+(e[i,2]-e[j,2])**2)
            if r==0:
                continue
            else:
                F_nn_x[i,j]=fc*np.abs(n[i,0]-n[j,0])/(r)**3
                F_nn_x[j,i]=fc*np.abs(n[i,0]-n[j,0])/(r)**3
                F_nn_y[i,j]=fc*np.abs(n[i,1]-n[j,1])/(r)**3
                F_nn_y[j,i]=fc*np.abs(n[i,1]-n[j,1])/(r)**3
                F_nn_z[i,j]=fc*np.abs(n[i,2]-n[j,2])/(r)**3
                F_nn_z[j,i]=fc*np.abs(n[i,2]-n[j,2])/(r)**3

    for i in range(len(F_ee_x)):
        for j in range(i, len(F_ee_x)):
            r=np.sqrt((e[i,0]-n[j,0])**2+(e[i,1]-n[j,1])**2+(e[i,2]-n[j,2])**2)
            if r==0:
                continue
            else:
                F_en_x[i,j]=-fc*(e[i,0]-n[j,0])/(r)**3
                F_en_x[j,i]=-fc*(e[i,0]-n[j,0])/(r)**3
                F_en_y[i,j]=-fc*(e[i,1]-n[j,1])/(r)**3
                F_en_y[j,i]=-fc*(e[i,1]-n[j,1])/(r)**3
                F_en_z[i,j]=-fc*(e[i,2]-n[j,2])/(r)**3
                F_en_z[j,i]=-fc*(e[i,2]-n[j,2])/(r)**3
    return F_ee_x, F_ee_y, F_ee_z, F_nn_x, F_nn_y, F_nn_z, F_en_x, F_en_y, F_en_z


rho=2
steps=13000
e=np.zeros((rho,3,steps))
n=np.zeros((rho,3,steps))
ve=np.zeros((rho,3,steps))
vn=np.zeros((rho,3,steps))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(rho):
    e[i,:,0]=np.random.uniform(-R,R,3)
    n[i,:,0]=np.random.uniform(-R,R,3)
    ve[i,:,0]=np.random.uniform(-V,V,3)
    # ve[i,1,0]=np.random.uniform(-V,V,1)
    # ve[i,2,0]=-(e[i,0,0]*ve[i,0,0]+e[i,1,0,0]*ve[i,1,0])/e[i,2,0,0]
    ax.scatter(e[i, 0, 0], e[i, 1, 0], e[i, 2, 0], color='k')
    ax.scatter(n[i, 0, 0], n[i, 1, 0], n[i, 2, 0], color='r')
    # ax.quiver(e[i, 0, 0], e[i, 1, 0], e[i, 2, 0], ve[i, 0, 0], ve[i, 1, 0], ve[i, 2, 0])

# plt.show()
def min_R(e,n):
    min_r=R
    n_i=0
    for i in range(len(n[:,0])):
        r=np.sqrt((e[0]-n[i,0])**2+(e[1]-n[i,1])**2+(e[2]-n[i,2])**2)
        if r<min_r:
            min_r=r
            n_i=i
    return min_r, n_i


def F_ex_e(I, e, n):
    F_x=0
    F_y=0
    F_z=0
    for i in range(len(e)):
        v_up=n[i]+np.array([0,0,2*R])
        v_dn=n[i]+np.array([0,0,-2*R])
        v_l=n[i]+np.array([0,-2*R,0])
        v_r=n[i]+np.array([0,2*R,0])
        v_f=n[i]+np.array([2*R,0,0])
        v_b=n[i]+np.array([-2*R,0,0])

        r_up=np.sqrt(np.sum((v_up-e[I])**2))
        F_x+=fc*(v_up[0]-e[I,0])/r_up**3

for t in range(1, steps):
    # dt=dt/2
    F_ee_x, F_ee_y, F_ee_z, F_nn_x, F_nn_y, F_nn_z, F_en_x, F_en_y, F_en_z=make_f(e[:,:,t-1],n[:,:,t-1])
    for i in range(rho):
        r, i_n=min_R(e[i,:,t-1],n[:,:,t-1])
        if r<2*10**(-10):
            e[i,0,t]=e[i,0,t-1]
            e[i,1,t]=e[i,1,t-1]
            e[i,2,t]=e[i,2,t-1]
            n[i_n,0,t]=n[i_n,0,t-1]
            n[i_n,1,t]=n[i_n,1,t-1]
            n[i_n,2,t]=n[i_n,2,t-1]
        else:
            e[i,0,t]=e[i,0,t-1]+ve[i,0,t-1]*dt+0.5*(np.sum(F_ee_x[i])+np.sum(F_en_x[i])-F_ex_e(i, e[:,:,t-1], n[:,:,t-1])[0])/me*(dt)**2
            e[i,1,t]=e[i,1,t-1]+ve[i,1,t-1]*dt+0.5*(np.sum(F_ee_y[i])+np.sum(F_en_y[i])-F_ex_e(i, e[:,:,t-1], n[:,:,t-1])[1])/me*(dt)**2
            e[i,2,t]=e[i,2,t-1]+ve[i,2,t-1]*dt+0.5*(np.sum(F_ee_z[i])+np.sum(F_en_z[i])-F_ex_e(i, e[:,:,t-1], n[:,:,t-1])[2])/me*(dt)**2
            n[i,0,t]=n[i,0,t-1]+vn[i,0,t-1]*dt+0.5*(np.sum(F_nn_x[i])+np.sum(F_en_x[:,i])+F_ex_n(i, e[:,:,t-1], n[:,:,t-1])[0])/mn*(dt)**2
            n[i,1,t]=n[i,1,t-1]+vn[i,1,t-1]*dt+0.5*(np.sum(F_nn_y[i])+np.sum(F_en_y[:,i])+F_ex_n(i, e[:,:,t-1], n[:,:,t-1])[1])/mn*(dt)**2
            n[i,2,t]=n[i,2,t-1]+vn[i,2,t-1]*dt+0.5*(np.sum(F_nn_z[i])+np.sum(F_en_z[:,i])+F_ex_n(i, e[:,:,t-1], n[:,:,t-1])[2])/mn*(dt)**2

            ve[i,0,t]=ve[i,0,t-1]+(np.sum(F_ee_x[i])+np.sum(F_en_x[i]))/me*(dt)
            ve[i,1,t]=ve[i,1,t-1]+(np.sum(F_ee_y[i])+np.sum(F_en_y[i]))/me*(dt)
            ve[i,2,t]=ve[i,2,t-1]+(np.sum(F_ee_z[i])+np.sum(F_en_z[i]))/me*(dt)
            vn[i,0,t]=vn[i,0,t-1]+(np.sum(F_nn_x[i])+np.sum(F_en_x[:,i]))/mn*(dt)
            vn[i,1,t]=vn[i,1,t-1]+(np.sum(F_nn_y[i])+np.sum(F_en_y[:,i]))/mn*(dt)
            vn[i,2,t]=vn[i,2,t-1]+(np.sum(F_nn_z[i])+np.sum(F_en_z[:,i]))/mn*(dt)

s=[]
d=[]
for i in range(rho):
    for t in range(1, steps):
        if t%10==0:
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            ax.scatter(e[i, 0, 0], e[i, 1, 0], e[i, 2, 0], color='k')
            ax.scatter(n[i, 0, 0], n[i, 1, 0], n[i, 2, 0], color='r')
            # ax.quiver(e[i, 0, 0], e[i, 1, 0], e[i, 2, 0], ve[i, 0, 0], ve[i, 1, 0], ve[i, 2, 0])
            # ax.quiver(e[i, 0, 0], e[i, 1, 0], e[i, 2, 0], n[i,0,0]-e[i, 0, 0], n[i,1,0]-e[i, 1, 0], n[i,2,0]-e[i, 2, 0])
            ax.quiver(e[i, 0, t-1], e[i, 1, t-1], e[i, 2, t-1], e[i,0,t]-e[i, 0, t-1], e[i,1,t]-e[i, 1, t-1], e[i,2,t]-e[i, 2, t-1], color='k')
            ax.quiver(n[i, 0, t-1], n[i, 1, t-1], n[i, 2, t-1], n[i,0,t]-n[i, 0, t-1], n[i,1,t]-n[i, 1, t-1], n[i,2,t]-n[i, 2, t-1], color='r')
            # plt.show()
            s.append(np.sqrt(np.sum((e[i,:,t]-e[i,:,t-1])**2)))
            d.append(np.sqrt(np.sum((e[i,:,t]-n[i,:,t-1])**2)))
# ax.set_xlim(-10,10)
# ax.set_ylim(-10,10)
# ax.set_zlim(-10,10)

plt.figure()
plt.plot(d, 'k.')
plt.yscale('log')
plt.title(np.amin(d))
plt.show()
