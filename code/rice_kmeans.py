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

mi_range = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]

train_x, train_y, test_x, test_y, num_classes, train_len = gen_rice(normalize=True)

subsample_rate = int(0.5*train_len)

rice_noise = {}
rice_acc = {}

for mi in mi_range:
    print(f"MI={mi}")
    est_noise = hybrid_noise_auto(train_x, train_y, run_kmeans, subsample_rate, eta=1e-6,
        num_classes = num_classes, max_mi=mi)
    rice_noise[mi] = est_noise

    with open(f'test_data/kmeans_rice_noise_mi={mi}.pkl', 'wb') as f:
        pickle.dump(rice_noise, f)
    print('rice noise complete')


    rice_num_features = len(train_x[0])
    num_trials = 1000

    avg_orig_acc = 0
    avg_priv_acc = 0
    orig_accs, priv_accs = [], []
    for i in range(num_trials):
        
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
        
        model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=None)
        predictions = model.predict(test_x)
        acc = accuracy_score(test_y, predictions)
        avg_orig_acc += acc
        for ind in range(len(cluster_centers)):
            c = np.random.normal(0, scale=rice_noise[mi][ind])
            cluster_centers[ind] += c
        new_centers = np.reshape(cluster_centers, (num_classes, rice_num_features))
        model.cluster_centers_ = new_centers
        new_predictions = model.predict(test_x)
        
        priv_acc = accuracy_score(test_y, new_predictions)
        avg_priv_acc += priv_acc
        orig_accs.append(acc)
        priv_accs.append(priv_acc)
    orig_acc_var = np.var(orig_accs)
    avg_orig_acc = np.mean(orig_accs)
    priv_acc_var = np.var(priv_accs)
    avg_priv_acc = np.mean(priv_accs)

    rice_acc[mi] = (avg_orig_acc, orig_acc_var, avg_priv_acc, priv_acc_var)

with open(f'test_data/kmeans_rice_auto_s=0.5_acc.pkl', 'wb') as f:
    pickle.dump(rice_acc, f)
print(mi, rice_acc)
