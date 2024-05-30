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
C_range = [0.05, 1.0]

with open(f'hybrid_svm/iris_svm_noise.pkl', 'rb') as f:
    all_noise = pickle.load(f) # keyed by MI, default is 0.5

subsample_rate = int(0.5*train_len)

num_trials = 1000
tpr = {}
fpr = {}

thresholds = [i / 100. for i in range(0, 101, 5)]

for add_noise in [True, False]:
    for mi in mi_range:
        for C_ind, C in enumerate(C_range):
            orig_noise, seed = all_noise[C]
            print(f'C={C}')
            for trial_ind in range(num_trials):
                false_dists = []
                print(f'trial number {trial_ind}')
                test_ind = np.random.choice(range(train_len))
                if add_noise:
                    false_dists = pickle.load(open(f'hybrid_lr/models/iris_svm_C={C}_noise={add_noise}_ind_{test_ind}_mi={mi}_dist.pkl', 'rb'))
                else:
                    false_dists = pickle.load(open(f'hybrid_lr/models/iris_svm_C={C}_noise={add_noise}_ind_{test_ind}_dist.pkl', 'rb'))
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
                
                    model, svm_vec = run_svm(shuffled_x1, shuffled_y1, num_classes=num_classes, seed=seed,
                                                 regularize=C)
                    if add_noise:
                        scaled_noise = {k: orig_noise[k] * (0.5 / mi) for k in orig_noise}
                        for ind in range(len(svm_vec)):
                            c = np.random.normal(0, scale=scaled_noise[ind])
                            svm_vec[ind] += c
                        reshape_val = num_classes
                        if num_classes == 2:
                            reshape_val = 1 # special case for binary
                        svm_mat = np.reshape(svm_vec, (reshape_val, num_features+1))
                        intercept = svm_mat[:, -1]
                        svm_mat = svm_mat[:, :-1]
                        model.coef_ = svm_mat
                        model.intercept_ = intercept


                    decision = model.decision_function([train_x[test_ind]])[0]
                    probs = [1. / (1. + np.exp(-1*x)) for x in decision]
                    norm_factor = sum(probs)
                    probs = [k / norm_factor for k in probs]
                    conf = max(probs)
                    # if we observe many scores geq conf, then likely false
                    # if we observe few, then likely true
                    normalized_score = len([i for i in false_dists if i >= conf]) / len(false_dists)
                    
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

                    model, svm_vec = run_svm(shuffled_x1, shuffled_y1, num_classes=num_classes, seed=seed,
                                                 regularize=C)

                    if add_noise:
                        scaled_noise = {k: orig_noise[k] * (0.5 / mi) for k in orig_noise}
                        for ind in range(len(svm_vec)):
                            c = np.random.normal(0, scale=scaled_noise[ind])
                            svm_vec[ind] += c
                        reshape_val = num_classes
                        if num_classes == 2:
                            reshape_val = 1 # special case for binary
                        svm_mat = np.reshape(svm_vec, (reshape_val, num_features+1))
                        intercept = svm_mat[:, -1]
                        svm_mat = svm_mat[:, :-1]
                        model.coef_ = svm_mat
                        model.intercept_ = intercept


                    decision = model.decision_function([train_x[test_ind]])[0]
                    probs = [1. / (1. + np.exp(-1*x)) for x in decision]
                    norm_factor = sum(probs)
                    probs = [k / norm_factor for k in probs]
                    conf = max(probs)
                    normalized_score = len([i for i in false_dists if i >= conf]) / len(false_dists)
                    
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
            with open(f'hybrid_lr/iris_svm_C={C}_mi={mi}_noise={add_noise}.pkl', 'wb') as f:
                pickle.dump(combined_dict, f)
