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
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from algs_lib import *
import sys


# LOAD cifar10
train_x, train_y, test_x, test_y, num_classes, train_len = gen_cifar10(normalize=True)
subsample_rate = int(0.5*train_len)

test_dims = list(range(1, 16))
test_dims = [1, 14]
mi_range = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625] # default noise is for mi = 0.5

num_trials = 1000

for mi in mi_range:
    acc_dict = {}
    for num_features in test_dims:
        with open(f'hybrid_data/cifar10_pca_noise_auto_dim={num_features}.pkl', 'rb') as f:
            orig_noise = pickle.load(f)
        scaled_noise = {k: orig_noise[k] * (0.5 / mi) for k in orig_noise}
        max_noise = max(scaled_noise.values())
        scaled_noise_iso = {k: max_noise for k in orig_noise}
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = shuffled_x1[:subsample_rate], shuffled_y1[:subsample_rate]
            model, components = run_pca(shuffled_x1, shuffled_y1, num_dims=num_features)
            predictions = model.inverse_transform(model.transform(test_x))

            acc = np.linalg.norm(test_x - predictions)
            acc /= np.linalg.norm(test_x)
            avg_orig_acc += acc
            shape = components.shape
            comp = components.flatten()
            for ind in range(len(comp)):
                c = np.random.normal(0, scale=scaled_noise_iso[ind])
                comp[ind] += c
            new_comp = np.reshape(comp, shape)
            model.components_ = new_comp
            new_predictions = model.inverse_transform(model.transform(test_x))

            priv_acc = np.linalg.norm(test_x - new_predictions)
            priv_acc /= np.linalg.norm(test_x)
            avg_priv_acc += priv_acc
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[num_features] = (avg_orig_acc, avg_priv_acc)

    print(mi, acc_dict)

    with open(f'hybrid_data/iso_cifar10_pca_dist_mi_{mi}.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)