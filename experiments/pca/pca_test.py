#encoding:utf8
import os, sys, pdb
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy as np
import math
from sklearn.decomposition import PCA


def gen_normal_data(N = 100):
    '''Generate Normal Distribution.'''
    u0 = 2
    delta0 = 2
    
    x = normal(u0, delta0, N)
    u1, delta1 = 2, 6

    y = normal(u1, delta1, N)

    mat1 = np.array(zip(x, y))
    

    theta = math.pi / 4.0
    mat_rot = np.array([
        [math.cos(theta), math.sin(theta)],
        [-math.sin(theta), math.cos(theta)]
        ])
    mat1 = mat1.dot(mat_rot)
    x, y = zip(*mat1)

    pca = do_pca(zip(x, y), 2)

    outX = pca.transform(zip(x, y))
    
    plt.plot(x, y, 'r.', alpha = 0.8, label = 'Original Distribution')
    x, y = zip(*outX)
    plt.plot(x, y, 'b.', alpha = 0.8, label = '2-dim pca re-projection')

    pca = do_pca(zip(x, y), 1)
    x1dim = pca.transform(zip(x, y))
    plt.plot(x1dim, len(x1dim) * [0, ], 'm.', alpha = 0.8, label = '1-dim projection')
    plt.hist(x1dim, bins = 30, weights = [0.1 ,] * len(x1dim), alpha = 0.4)
    

    plt.xlim((-20, 30))
    plt.ylim((-20, 30))
    plt.grid(True)
    plt.legend(numpoints = 1)
    plt.show()


def do_pca(x, n_components):
    '''PCA test.'''
    pca = PCA(n_components = n_components, )
    pca.fit(x)
    return pca

if __name__ == '__main__':
    gen_normal_data(1000)
