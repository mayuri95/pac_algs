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

# bean

train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True)

C_vals = [1.0, 0.05]
dims = [1, 8]

num_trials = 1000
subsample_rate = train_len

baseline_accs = {}

rebalance = [True, False]
# K MEANS
baseline_accs['kmeans'] = {}
for reb in rebalance:
    avg_acc = 0
    for i in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
        
        model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=None, rebalance=reb)
        predictions = model.predict(test_x)
        acc = accuracy_score(test_y, predictions)
        avg_acc += acc
    avg_acc /= num_trials
    baseline_accs['kmeans'][reb] = avg_acc

# SVM
baseline_accs['svm'] = {}
for C in C_vals:
    avg_acc = 0
    for i in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
        model, svm_vec = run_svm(shuffled_x1, shuffled_y1, num_classes=num_classes, seed=None,
                                 regularize=C)
        acc = model.score(test_x, test_y)
        avg_acc += acc
    avg_acc /= num_trials
    baseline_accs['svm'][C] = avg_acc

# PCA

baseline_accs['pca'] = {}
for dim in dims:
    avg_acc = 0
    for i in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = shuffled_x1[:subsample_rate], shuffled_y1[:subsample_rate]
        model, components = run_pca(shuffled_x1, shuffled_y1, num_dims=dim)
        predictions = model.inverse_transform(model.transform(test_x))
        acc = np.linalg.norm(test_x - predictions)
        acc /= np.linalg.norm(test_x)
        avg_acc += acc
    avg_acc /= num_trials
    baseline_accs['pca'][dim] = avg_acc

with open('hybrid_baseline/bean_baselines.pkl', 'wb') as f:
	pickle.dump(baseline_accs, f)


