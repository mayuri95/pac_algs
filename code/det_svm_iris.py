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


train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True)

subsample_rate = int(0.25*train_len)

C_range = [x / 100. for x in range(1, 101)]
num_trials = 1000

for mi in mi_range:
    iris_noise = {}
    iris_acc = {}
    for C in C_range:
        print(f"C={C}, mi={mi}")
        
        est_noise = det_mechanism_noise_auto(train_x, train_y, run_svm, subsample_rate, eta=1e-3,
                                         num_classes = num_classes, regularize=C, max_mi=mi)
        iris_noise[C] = est_noise

        num_features = len(train_x[0])
        model = svm.LinearSVC(dual=False, random_state=None)
        model.fit(train_x[:50], train_y[:50])
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
            model, svm_vec = run_svm(shuffled_x1, shuffled_y1, num_classes=num_classes, seed=None,
                                     regularize=C)
            acc = model.score(test_x, test_y)
            avg_orig_acc += acc
            mean_noise = [0]*len(svm_vec)
            noise_vec = np.random.multivariate_normal(mean_noise, iris_noise[C])
            for ind in range(len(svm_vec)):
                svm_vec[ind] += noise_vec[ind]
            reshape_val = num_classes
            if num_classes == 2:
                reshape_val = 1 # special case for binary
            svm_mat = np.reshape(svm_vec, (reshape_val, num_features+1))
            intercept = svm_mat[:, -1]
            svm_mat = svm_mat[:, :-1]
            model.coef_ = svm_mat
            model.intercept_ = intercept
            priv_acc = model.score(test_x, test_y)
            avg_priv_acc += priv_acc
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        iris_acc[C] = (avg_orig_acc, avg_priv_acc)
        print(f'acc={avg_orig_acc}, {avg_priv_acc}')

    with open(f'test_data/iris_svm_aniso_auto_s=0.25_acc_mi={mi}.pkl', 'wb') as f:
        pickle.dump(iris_acc, f)
        
    with open(f'test_data/iris_svm_aniso_auto_s=0.25_noise_mi={mi}.pkl', 'wb') as f:
        pickle.dump(iris_noise, f)
