import numpy as np
import matplotlib.pyplot as plt

a=np.array([[1,2,3,2,1], [12,13,12,11,10]]).T
print(np.roll(a, -np.argmax(a, axis=0), axis=[0,0]))
