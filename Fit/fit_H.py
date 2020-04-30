import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf

pmt=4
path='/home/gerak/Desktop/DireXeno/WFsim/PMT{}/'.format(pmt)
Data=np.load(path+'H.npz')
H=Data['H']
spec_x=Data['spec_x']
spec_y=Data['spec_y']
