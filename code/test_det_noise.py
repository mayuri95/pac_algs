from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
import pickle
import os
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.datasets import make_blobs
import math
import argparse
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo 
from collections import Counter
from scipy.stats import entropy
from sklearn import tree
from sklearn import preprocessing
from itertools import product
from algs_lib import *

def run_kmeans_fixed_seed(train_x, train_y, num_clusters):
    rand_state = np.random.RandomState(0)
    model = KMeans(n_clusters=num_clusters, random_state=rand_state, n_init="auto").fit(train_x)
    
    centers = model.cluster_centers_
    assert len(centers) == num_clusters
    mapping = infer_cluster_labels(model, train_y)
    
    ordered_centers = []
    new_centers = []
    
    for ind in range(len(centers)):
        index = mapping[ind]
        curr_center = centers[index]
        new_centers.append(curr_center)
        for k in curr_center:
            ordered_centers.append(k)
    model.cluster_centers_ = np.array(new_centers)
    return model, ordered_centers

def calc_cov_large_gap(d, c, v, beta, eigs, u):
    sigma_matrix = np.zeros((d, d))
    for i in range(d):
        num = 2.*v
        # print('num', num)
        denom_init = (eigs[i])**0.5
        denom_second = 0.
        # print('denom', denom_init)
        for k in range(d):
            denom_second += (eigs[k] )**0.5
            # print('const is ', 10*c*v/beta)
            # print('denom second', denom_second)
        denom = denom_init * denom_second
        sigma_matrix[i][i] = num/denom
            # print(sigma_matrix[i][i])
    # print('sigma inverse', np.linalg.inv(sigma_matrix).diagonal())
    noise_matrix =  np.matmul(
        np.matmul(u, np.linalg.inv(sigma_matrix)), u.T)

    return noise_matrix

def det_noise_anisotropic(train_x, train_y, mechanism, subsample_rate, num_classes=None, prefix=None, max_mi = 1.):

    sec_v = max_mi / 2
    sec_beta = max_mi - sec_v
    r = calc_r(train_x)
    gamma = 0.01
    c = 0.001
    num_trials = 1000
    avg_dist = 0.

    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes
    est_y = []
    
    for _ in range(num_trials):
        shuffled_x, shuffled_y = shuffle(train_x, train_y)
        
        
        shuffled_x, shuffled_y = get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate)

        est_y.append(mechanism(shuffled_x, shuffled_y, num_classes)[1])

    y_mean = np.average(est_y, axis=0)
    y_cov = np.cov(np.array(est_y).T)
    d = len(est_y[0]) # length of flattened vector
    
    u, eigs, v = np.linalg.svd(y_cov)
    cov = calc_cov_large_gap(d, c, sec_v, sec_beta, eigs, u)
    return cov

def det_noise_isotropic(train_x, train_y, mechanism, subsample_rate, num_classes=None, prefix=None, max_mi = 1.):

    sec_v = max_mi / 2
    sec_beta = max_mi - sec_v
    r = calc_r(train_x)
    gamma = 0.01
    c = 0.001
    num_trials = 1000
    avg_dist = 0.

    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes
    est_y = []
    
    for _ in range(num_trials):
        shuffled_x, shuffled_y = shuffle(train_x, train_y)
        
        
        shuffled_x, shuffled_y = get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate)

        est_y.append(mechanism(shuffled_x, shuffled_y, num_classes)[1])

    y_mean = np.average(est_y, axis=0)
    y_cov = np.cov(np.array(est_y).T)
    d = len(est_y[0]) # length of flattened vector
    print('iso d is ', d)
    
    u, eigs, v = np.linalg.svd(y_cov)
    cov = calc_cov_small_gap(d, c, sec_v, eigs)
    return cov



def gen_syn_high_dim(dim, log_num_clusters=2, num_train=10000, num_test=3000, normalize=False):
    cluster_std = 0.2
    num_test_samples = num_test
    items = [-1, 1]

    centers = []
    for item in product(items, repeat=log_num_clusters):
        centers.append(list(item))
    assert dim >= len(centers)
    remaining_dim = dim - log_num_clusters
    syn_num_classes = len(centers)


    syn_train_samples = num_train
    syn_num_classes = len(centers)
    syn_train_x, syn_train_y, centers = make_blobs(n_samples=syn_train_samples,
                                           centers=centers, n_features=dim, random_state=0,
                                           cluster_std=cluster_std, return_centers=True)

    syn_test_x, syn_test_y = make_blobs(n_samples=num_test_samples, centers=centers, n_features=dim,
                                random_state=0, cluster_std=cluster_std)

    train_zeros = np.zeros((len(syn_train_x), 1))  # zeros column as 2D array
    test_zeros = np.zeros((len(syn_test_x), 1))  # zeros column as 2D array
    for _ in range(remaining_dim):
        syn_train_x = np.hstack((syn_train_x, train_zeros))
        syn_test_x = np.hstack((syn_test_x, test_zeros))

    return syn_train_x, syn_train_y, syn_test_x, syn_test_y, syn_num_classes, syn_train_samples

covs = {}
dims = [2**d for d in range(2,7)]
# dims = [4, 64]
for dim in dims:
    train_x, train_y, test_x, test_y, num_classes, train_len = gen_syn_high_dim(dim)
    # print(len(centers))
    subsample_rate = int(0.5*train_len)
    aniso_cov_mat = det_noise_anisotropic(train_x, train_y, run_kmeans_fixed_seed, subsample_rate, num_classes = num_classes, max_mi=1)[2]
    iso_cov_mat = det_noise_isotropic(train_x, train_y, run_kmeans_fixed_seed, subsample_rate, num_classes = num_classes, max_mi=1)[2]
    print(dim, np.linalg.norm(aniso_cov_mat, ord=2), np.linalg.norm(iso_cov_mat, ord=2))
    covs[dim] = (iso_cov_mat, aniso_cov_mat)
    print(f'completed dim {dim}')

with open(f'data_0120/syn_data_high_dim.pkl', 'wb') as f:
    pickle.dump(covs, f)
        
