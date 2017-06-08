#encoding:utf8

# MCMC Model
import os
import sys
import pymc
import pdb
import math
import scipy.signal
import matplotlib.pyplot as plt
import scipy.signal as signal
from pymc import DiscreteUniform, Exponential, deterministic, Poisson, Uniform
import numpy as np
import json

# Hermit functions
HermitFunction_max_level = 8
def HermitFunction(level, size):
    '''Return hermit function for P wave.'''
    size = int(size)
    if size < 0:
        raise Exception('Size must be greater or equal to zero!')

    def He0(x):
        return 1.0
    def He1(x):
        return x
    def He2(x):
        return x * x - 1
    def He3(x):
        return x ** 3.0 - 3.0 * x
    def He4(x):
        return x ** 4.0 - 6.0 * x ** 2.0 + 3.0
    def He5(x):
        return x ** 5.0 - 10.0 * x ** 3.0 + 15.0 * x
    def He6(x):
        return x ** 6.0 - 15.0 * x ** 4.0 + 45.0 * x ** 2 - 15.0
    def He7(x):
        return x ** 7.0 - 21.0 * x ** 5.0 + 105.0 * x ** 3 - 105.0 * x
    # Mapping wave_width to range [-3,3]
    x_ratio = 6.0 / size 
    if level == 0:
        hermit = He0
    elif level == 1:
        hermit = He1
    elif level == 2:
        hermit = He2
    elif level == 3:
        hermit = He3
    elif level == 4:
        hermit = He4
    elif level == 5:
        hermit = He5
    elif level == 6:
        hermit = He6
    elif level == 7:
        hermit = He7

    data = [hermit((x - size / 2) * x_ratio) / 20.0 for x in xrange(0, size)]
    
    return np.array(data)


def GetBaselineMatrix(signal_length, fs):
    '''Get baseline coefficient matrix.(0.5Hz~1Hz)'''
    mat = [[1.0,] * signal_length]
    # 0.5Hz
    sin_list = [math.sin(x / fs * math.pi) for x in xrange(0, signal_length)]
    mat.append(sin_list)
    cos_list = [math.cos(x / fs * math.pi) for x in xrange(0, signal_length)]
    mat.append(sin_list)
    # 1Hz
    sin_list = [math.sin(x / fs * math.pi * 2.0) for x in xrange(0, signal_length)]
    mat.append(sin_list)
    cos_list = [math.cos(x / fs * math.pi * 2.0) for x in xrange(0, signal_length)]
    mat.append(sin_list)
    return np.array(mat)


def MakeModel(sig1, max_hermit_level = 7):
    '''Create P wave delineation model for MCMC.'''
    # Load ECG segment array
    sig1 = np.array(sig1, dtype = np.float32)

    # Length of the ECG segment 
    len_sig1 = sig1.size

    common_length = len_sig1
    
    hermit_coefs = list()
    # Dc baseline
    # coef = pymc.Normal('hc0', mu = 0, tau = 0.003)
    coef = DiscreteUniform('hc0', lower=-0.05,
            upper= 0.05, doc='hc0')
    hermit_coefs.append(coef)
    # level 1
    ind = 1
    hermit_coefs.append(pymc.Normal('hc%d' % ind, mu = 0, tau = 1))
    # level 2
    ind = 2
    hermit_coefs.append(pymc.Normal('hc%d' % ind, mu = 0, tau = 1))
    for ind in xrange(3, HermitFunction_max_level):
        coef = pymc.Normal('hc%d' % ind, mu = 0, tau = 1)
        hermit_coefs.append(coef)

    
    @deterministic(plot=False)
    def wave_diff(
            hermit_coefs = hermit_coefs,
            ):
        ''' Concatenate wave.'''

        out = sig1[:common_length]
        fitting_curve = np.zeros(common_length,)
        for level, coef in zip(xrange(0, max_hermit_level), hermit_coefs):
            fitting_curve += HermitFunction(level, int(common_length)) * coef
        
        return out - fitting_curve


    diff_sig = pymc.Normal('diff_sig',
            mu = wave_diff,
            tau = 17,
            value = [0,] * common_length,
            observed = True)

    return locals()
