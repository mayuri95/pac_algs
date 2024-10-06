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
num_features = len(train_x[0])

mi_range = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

subsample_rate = int(0.5*train_len)
seed = 743895091


with open(f'data/iris_kmeans_noise_rebalance=False.pkl', 'rb') as f:
    orig_noise, xs = pickle.load(f)[0.5] # keyed by MI, default is 0.5
max_noise = max(orig_noise.values())

num_trials = 1000
tpr = {}
fpr = {}

thresholds = [i / 100. for i in range(0, 101, 5)]

for add_noise in [True, False]:
    for mi in mi_range:
        scaled_noise = {k: max_noise * (0.5 / mi) for k in orig_noise}
        for trial_ind in range(num_trials):
            false_dists = []


            print(f'trial number {trial_ind}')
            test_ind = np.random.choice(range(train_len))
            keys_in = [k for k in xs if test_ind in xs[k]]
            keys_out = [k for k in xs if test_ind not in xs[k]]
            if add_noise:
                false_dists = pickle.load(open(f'hybrid_lr/models/iris_kmeans_iso_noise={add_noise}_ind_{test_ind}_mi={mi}_dist.pkl', 'rb'))
            else:
                false_dists = pickle.load(open(f'hybrid_lr/models/iris_kmeans_iso_noise={add_noise}_ind_{test_ind}_dist.pkl', 'rb'))
            
            # true positive test
            choice = np.random.choice(keys_in)

            shuffled_x1, shuffled_y1 = train_x[xs[choice]], train_y[xs[choice]]

            for t in thresholds:

                actual = True
            
                model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=seed)
                
                if add_noise:
                    for ind in range(len(cluster_centers)):
                        c = np.random.normal(0, scale=scaled_noise[ind])
                        cluster_centers[ind] += c
                    new_centers = np.reshape(cluster_centers, (num_classes, num_features))
                    model.cluster_centers_ = new_centers

                cluster_centers = np.array(cluster_centers).reshape((num_classes, -1))

                cluster_id = model.predict([train_x[test_ind]])[0]

                all_dists = []
                num_clusters = len(cluster_centers)
                for cid in range(num_clusters):

                    all_dists.append(np.linalg.norm(train_x[test_ind] - cluster_centers[cid]))
                norm_factor = sum(all_dists)
                all_dists = [k/norm_factor for k in all_dists]
                dist = 1-min(all_dists)
                # if we observe many scores geq conf, then likely false
                # if we observe few, then likely true
                normalized_score = len([i for i in false_dists if i >= dist]) / len(false_dists)
                
                if normalized_score >= t:
                    guess= False
                else:
                    guess = True
                if t not in tpr:
                    tpr[t] = 0
                if actual == guess:

                    tpr[t] += 1

            # false positive test
            choice = np.random.choice(keys_out)

            shuffled_x1, shuffled_y1 = train_x[xs[choice]], train_y[xs[choice]]

            actual = False
            for t in thresholds:

                model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=seed)
                
                if add_noise:
                    for ind in range(len(cluster_centers)):
                        c = np.random.normal(0, scale=scaled_noise[ind])
                        cluster_centers[ind] += c
                    new_centers = np.reshape(cluster_centers, (num_classes, num_features))
                    model.cluster_centers_ = new_centers

                cluster_centers = np.array(cluster_centers).reshape((num_classes, -1))

                cluster_id = model.predict([train_x[test_ind]])[0]

                all_dists = []
                num_clusters = len(cluster_centers)
                for cid in range(num_clusters):

                    all_dists.append(np.linalg.norm(train_x[test_ind] - cluster_centers[cid]))
                norm_factor = sum(all_dists)
                all_dists = [k/norm_factor for k in all_dists]
                dist = 1-min(all_dists)
                # if we observe many scores geq conf, then likely false
                # if we observe few, then likely true
                normalized_score = len([i for i in false_dists if i >= dist]) / len(false_dists)
                        
                if normalized_score >= t:
                    guess= False
                else:
                    guess = True
                if t not in fpr:
                    fpr[t] = 0
                if actual != guess:

                    fpr[t] += 1
        for t in tpr:
            tpr[t] /= num_trials
        for t in fpr:
            fpr[t] /= num_trials

        combined_dict = {}
        for t in tpr:
            combined_dict[t] = (tpr[t], fpr[t])
        with open(f'hybrid_lr/iris_kmeans_iso_mi={mi}_noise={add_noise}.pkl', 'wb') as f:
            pickle.dump(combined_dict, f)
