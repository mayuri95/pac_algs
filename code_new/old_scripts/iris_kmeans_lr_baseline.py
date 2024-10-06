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

noise_dicts = {4.0: 0.05749397913745031, 1.0: 0.05861768660841647, 0.25: 0.13931298872701633, 0.0625: 0.1370552116845034, 0.015625: 0.12594080321957227}
mi = 1./float(sys.argv[1])
print(mi)

noise_val = noise_dicts[mi]
train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True)

num_features = len(train_x[0])

subsample_rate = int(0.5*train_len)


num_trials = 1000
tpr = {}
fpr = {}

thresholds = [i / 100. for i in range(0, 101, 5)]

for add_noise in [False]:
    for trial_ind in range(num_trials):
        print(f'trial {trial_ind}')
        test_ind = np.random.choice(range(train_len))

        false_dists = pickle.load(open(f'data_0120/models/iris_kmeans_ind_{test_ind}_dist.pkl', 'rb'))
        # true positive test
        other_x = np.delete(train_x, test_ind, 0)
        other_y = np.delete(train_y, test_ind, 0)

        shuffled_x1, shuffled_y1 = shuffle(other_x, other_y)
        num_samples_true = subsample_rate-1
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, num_samples_true)


        shuffled_x1 = np.append(shuffled_x1, np.atleast_2d(train_x[test_ind]), axis=0)
        shuffled_y1 = np.append(shuffled_y1, np.atleast_1d(train_y[test_ind]), axis=0)
        for t in thresholds:

            actual = True

            model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=None)
            
            if add_noise:
                for ind in range(len(cluster_centers)):
                    c = np.random.normal(0, scale=noise_val)
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
        other_x = np.delete(train_x, test_ind, 0)
        other_y = np.delete(train_y, test_ind, 0)

        shuffled_x1, shuffled_y1 = shuffle(other_x, other_y)

        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)


        actual = False
        for t in thresholds:

            model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=None)
            
            if add_noise:
                for ind in range(len(cluster_centers)):
                    c = np.random.normal(0, scale=noise_val)
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
    print(combined_dict)
    with open(f'data_0120/iris_kmeans_mi={mi}_noise={add_noise}_baseline.pkl', 'wb') as f:
        pickle.dump(combined_dict, f)
