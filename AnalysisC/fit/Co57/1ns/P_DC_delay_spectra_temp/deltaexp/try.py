import numpy as np

H=np.random.rand(5,6)
H=H/np.sum(H, axis=0)
print(H, np.sum(H[:,0]))

for pmt in range(len(H[0])):
     
