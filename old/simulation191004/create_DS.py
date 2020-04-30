from classes import WF, Event, DataSet
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from fun import do_smd
from shootPEs import shootPEs, show_event, find_posible_angels, find_angels
from hit_fun_order import find_peaks, Reconstruct_WF
from scipy import special
import pickle
import pandas as pd

PMT_num=20
time_samples=1024
label='acisotropic_center'
j=0
id=0
dt=0.5
good_chns=[0,1,2,3,4,5,6,7,8,9,10,11,14,15,16,17,19]

dataset=DataSet(label+'_{}'.format(j), 1)
dataset.first_event=id
find_posible_angels(dataset)
start_time = time.time()
while id<1000:
    if id%1==0:
        print('Event number {} ({} files per sec).'.format(id, 1/(time.time()-start_time)))
        start_time = time.time()
    PE_num=int(np.random.normal(10000, 500, 1)[0])
    while PE_num<0:
        PE_num=int(np.random.normal(10000, 500, 1)[0])
    t, pmt = shootPEs(PE_num, 1, 'dont_plot')
    event = Event(id)
    for chn in good_chns:
        wf=WF(chn)
        wf.pmt=chn
        bins=np.linspace(0,200, 101)
        if not len(bins)==100+1:
            print('problem with the binning')
            sys.exit()
        if len(pmt)>0:
            h, bins=np.histogram(t[np.nonzero(pmt==chn)[0]], bins=bins)
            wf.PE=h
            wf.T=0.5*(bins[:-1]+bins[1:])
        event.wf.append(wf)
        event.PEs+=np.sum(wf.PE)
    event.dt=dt
    find_angels(event, dataset.posible_angles)
    dataset.events.append(event)
    id+=1

    if 'dont_plot'=='plot':
        show_event(event, dataset.posible_angles)


    if len(dataset.events)>500:
        dataset.last_event=id-1
        with open('190809sim/'+label+'_events{}to{}'.format(dataset.first_event, dataset.last_event), 'wb') as output:
            pickle.dump(dataset, output)
            j+=1
            dataset=DataSet(label+'_{}'.format(j), 1)
            dataset.first_event=id
if len(dataset.events)>0:
    dataset.last_event=id-1
    with open('190809sim/'+label+'_events{}to{}'.format(dataset.first_event, dataset.last_event), 'wb') as output:
        pickle.dump(dataset, output)
        j+=1
        dataset=DataSet(label+'_{}'.format(j), 1)
        dataset.first_event=id
