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

train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True)

subsample_rate = int(0.5*train_len)

acc_dict = {}

num_features = len(train_x[0])
num_trials = 1000
seed = 743895091


mi_range = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 1/128.] # default noise is for mi = 0.5


for rebalance in [True, False]:
    with open(f'data/iris_kmeans_noise_rebalance={rebalance}.pkl', 'rb') as f:
        orig_noise, xs = pickle.load(f)[0.5] # keyed by MI, default is 0.5
    for mi in mi_range:
        avg_orig_acc = 0
        avg_priv_acc = 0
        orig_accs, priv_accs = [], []
        scaled_noise = {k: orig_noise[k] * (0.5 / mi) for k in orig_noise}
        for i in range(num_trials):
            
            choice = np.random.choice(range(1024))
            shuffled_x1, shuffled_y1 = train_x[xs[choice]], train_y[xs[choice]]
            
            model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=seed, rebalance=rebalance)
            predictions = model.predict(test_x)
            acc = accuracy_score(test_y, predictions)
            avg_orig_acc += acc
            for ind in range(len(cluster_centers)):
                c = np.random.normal(0, scale=scaled_noise[ind])
                cluster_centers[ind] += c
            new_centers = np.reshape(cluster_centers, (num_classes, num_features))
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
        print(avg_orig_acc, avg_priv_acc)

        acc_dict[mi] = (avg_orig_acc, orig_acc_var, avg_priv_acc, priv_acc_var)
    with open(f'data/iris_kmeans_acc_bal={rebalance}.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)

for rebalance in [True, False]:
    with open(f'data/iris_kmeans_noise_rebalance={rebalance}.pkl', 'rb') as f:
        orig_noise, xs = pickle.load(f)[0.5] # keyed by MI, default is 0.5
    for mi in mi_range:
        avg_orig_acc = 0
        avg_priv_acc = 0
        orig_accs, priv_accs = [], []
        scaled_noise = {k: orig_noise[k] * (0.5 / mi) for k in orig_noise}
        max_noise = max(scaled_noise.values())
        scaled_noise = {k: max_noise for k in scaled_noise}
        for i in range(num_trials):
            
            choice = np.random.choice(range(1024))
            shuffled_x1, shuffled_y1 = train_x[xs[choice]], train_y[xs[choice]]
            
            model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=seed, rebalance=rebalance)
            predictions = model.predict(test_x)
            acc = accuracy_score(test_y, predictions)
            avg_orig_acc += acc
            for ind in range(len(cluster_centers)):
                c = np.random.normal(0, scale=scaled_noise[ind])
                cluster_centers[ind] += c
            new_centers = np.reshape(cluster_centers, (num_classes, num_features))
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
        print(avg_orig_acc, avg_priv_acc)

        acc_dict[mi] = (avg_orig_acc, orig_acc_var, avg_priv_acc, priv_acc_var)
    with open(f'data/iris_iso_kmeans_acc_bal={rebalance}.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)