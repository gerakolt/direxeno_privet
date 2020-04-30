import numpy as np
import matplotlib.pyplot as plt

path='/home/gerak/Desktop/DireXeno/190803/Co57/'
rec=np.load(path+'height.npz')['rec']

plt.figure()
plt.title('Co57')
plt.hist(rec['h'][:,0], bins=50, histtype='step', linewidth=5)
plt.hist(rec['h'][:,1], bins=50, histtype='step', linewidth=5)
plt.xlabel('Signal height in units of mean SPE height', fontsize=25)
plt.show()
