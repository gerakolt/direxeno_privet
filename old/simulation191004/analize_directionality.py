from classes import WF, Event, DataSet
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from fun import do_smd
from shootPEs import shootPEs, show_event, find_posible_angels, find_angels, Dipol
from hit_fun_order import find_peaks, Reconstruct_WF
from scipy import special
import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join
from mpl_toolkits.mplot3d import Axes3D

path='./190809sim/'
files = [path+f for f in listdir(path) if isfile(join(path, f))]
fig = plt.figure(figsize=[15,10])
ax0=fig.add_subplot(2,1,1)
ax1=fig.add_subplot(2,1,2)
color=['y','r']
i=0
for file in files:
    if file.endswith('.pkl'):
            continue
    with open(file, 'rb') as input:
        D=[]
        dataset = pickle.load(input)
        for event in dataset.events:
            dipol, PMT = Dipol(event)
            D.append(dipol)
            print('in event', event.id, 'number of PEs', event.PEs)
            ax0.plot(dataset.posible_angles, event.angles/event.PEs, color[i]+'o')
        i+=1
        ax1.hist(D, bins=25, histtype='step', label=file[-30:-18])
        ax1.set_xlabel('dipol')
        ax1.legend()
plt.show()
