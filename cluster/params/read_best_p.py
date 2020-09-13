import numpy as np
name='best_p_Cs137'
data=np.load(name+'.npz')
data0=np.load(name+'0.npz')

print(name)
if data['l_min']<data0['l_min']:
    print(data['p'])
    print(data['l_min'])
else:
    print(data['p'])
    print(data['l_min'])
