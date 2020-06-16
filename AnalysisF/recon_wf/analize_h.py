import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes, get_delays
import sys

data=np.load('h.npz')
rec=data['rec']
for i in range(6):
    plt.figure()
    plt.hist(rec['h'][:,i], bins=100)
    plt.show()
