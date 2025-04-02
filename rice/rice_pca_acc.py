from sklearn.utils import shuffle
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
import sys

dims = [1,6]

mis = [1/2**(x) for x in range(-2, 8)]
norm = 'minmax'
num_trials = 1000
print(mis)

for mi in mis:
    acc_dict = {}
    for C_i, dim in enumerate(dims):
        with open(f'noise/rice_pca_noise_auto_dim={dim}.pkl', 'rb') as f:
            all_noise = pickle.load(f)
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_rice(normalize=True, norm_kind =norm)

        subsample_rate = int(0.5*train_len)
        orig_noise, seed = all_noise[dim]
        rand_state = np.random.RandomState(seed)
        scaled_noise = {k: np.sqrt(orig_noise[k] * (0.5 / mi)) for k in orig_noise if k != 's1'}
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            print(i)
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = shuffled_x1[:subsample_rate], shuffled_y1[:subsample_rate]
            s1 = orig_noise['s1']
            model, components = run_pca(shuffled_x1, shuffled_y1, num_dims=dim, seed=seed)
            components = transform(s1, components)
            predictions = model.inverse_transform(model.transform(test_x))

            acc = np.linalg.norm(test_x - predictions)
            acc /= np.linalg.norm(test_x)
            avg_orig_acc += acc
            shape = components.shape
            comp = components.flatten()
            for ind in range(len(comp)):
                c = np.random.normal(0, scale=scaled_noise[ind])
                comp[ind] += c
            new_comp = np.reshape(comp, shape)
            model.components_ = new_comp
            new_predictions = model.inverse_transform(model.transform(test_x))

            priv_acc = np.linalg.norm(test_x - new_predictions)
            priv_acc /= np.linalg.norm(test_x)
            avg_priv_acc += priv_acc
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[dim] = (avg_orig_acc, avg_priv_acc)
        print(f'dim={dim}, mi={mi}, norm={norm}, acc={avg_orig_acc}, {avg_priv_acc}')
    with open(f'accs/rice_pca_dist_mi_{mi}.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)

for mi in mis:
    acc_dict = {}
    for C_i, dim in enumerate(dims):
        with open(f'noise/rice_pca_noise_auto_dim={dim}.pkl', 'rb') as f:
            all_noise = pickle.load(f)
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_rice(normalize=True, norm_kind =norm)

        subsample_rate = int(0.5*train_len)
        orig_noise, seed = all_noise[dim]
        rand_state = np.random.RandomState(seed)
        scaled_noise = {k: np.sqrt(orig_noise[k] * (0.5 / mi)) for k in orig_noise if k != 's1'}
        max_noise = max(scaled_noise.values())
        scaled_noise = {k: max_noise for k in orig_noise}
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = shuffled_x1[:subsample_rate], shuffled_y1[:subsample_rate]
            s1 = orig_noise['s1']
            model, components = run_pca(shuffled_x1, shuffled_y1, num_dims=dim, seed=seed)
            components = transform(s1, components)
            predictions = model.inverse_transform(model.transform(test_x))

            acc = np.linalg.norm(test_x - predictions)
            acc /= np.linalg.norm(test_x)
            avg_orig_acc += acc
            shape = components.shape
            comp = components.flatten()
            for ind in range(len(comp)):
                c = np.random.normal(0, scale=scaled_noise[ind])
                comp[ind] += c
            new_comp = np.reshape(comp, shape)
            model.components_ = new_comp
            new_predictions = model.inverse_transform(model.transform(test_x))

            priv_acc = np.linalg.norm(test_x - new_predictions)
            priv_acc /= np.linalg.norm(test_x)
            avg_priv_acc += priv_acc
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[dim] = (avg_orig_acc, avg_priv_acc)
        print(f'dim={dim}, mi={mi}, norm={norm}, acc={avg_orig_acc}, {avg_priv_acc}')
    with open(f'accs/rice_iso_pca_dist_mi_{mi}.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)
