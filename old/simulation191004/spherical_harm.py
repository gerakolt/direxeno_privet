import scipy as sci
import scipy.special as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
#http://balbuceosastropy.blogspot.com/2015/06/spherical-harmonics-in-python.html
l1=4
m1=2
l2=5
m2=2
PHI, THETA = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
Norm = np.abs(0.8*sp.sph_harm(m1, l1, PHI, THETA)+0.2*sp.sph_harm(m2, l2, PHI, THETA)) #Array with the absolute values of Ylm
R = (0.8*sp.sph_harm(m2, l2, PHI, THETA)+0.2*sp.sph_harm(m1, l1, PHI, THETA)).real #Array with the real values of Ylm
# print(R, type(R))
# sys.exit()

X = Norm * np.sin(THETA) * np.cos(PHI)
Y = Norm * np.sin(THETA) * np.sin(PHI)
Z = Norm * np.cos(THETA)


N = Norm/Norm.max()    # Normalize R for the plot colors to cover the entire range of colormap.
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12,10))
im = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(N))
ax.set_title(r'$|Y^2_ 4|$', fontsize=20)
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(Norm)    # Assign the unnormalized data array to the mappable
                  #so that the scale corresponds to the values of R
fig.colorbar(m, shrink=0.8);
########################### The color represents |Y| in (X,Y,Z)


X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
Z = R * np.cos(THETA)


norm = colors.Normalize()
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(14,10))
m = cm.ScalarMappable(cmap=cm.jet)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(norm(R)))
ax.set_title('real$(Y^2_ 4)$', fontsize=20)
m.set_array(R)
fig.colorbar(m, shrink=0.8);

###########################################

s=1
X = (s*R+1) * np.sin(THETA) * np.cos(PHI)
Y = (s*R+1) * np.sin(THETA) * np.sin(PHI)
Z = (s*R+1) * np.cos(THETA)


norm = colors.Normalize()
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(14,10))
m = cm.ScalarMappable(cmap=cm.jet)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(norm(R)))
ax.set_title('1 + real$(Y^2_ 4)$', fontsize=20)
m.set_array(R)
fig.colorbar(m, shrink=0.8);


plt.show()
