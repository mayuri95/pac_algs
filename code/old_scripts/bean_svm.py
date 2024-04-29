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

mi = 1./eval(sys.argv[1])
print(f'mi = {mi}')
mi_range = [mi]

train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True)

subsample_rate = int(0.5*train_len)

C_range = [x / 100 for x in range(1, 101)]

num_trials = 1000

for mi in mi_range:
    bean_noise = {}
    bean_acc = {}
    for C in C_range:
        print(f"C={C}")
        
        est_noise = hybrid_noise_auto(train_x, train_y, run_svm, subsample_rate, eta=1e-6,
            num_classes = num_classes, max_mi=mi, regularize=C)
        bean_noise[C] = est_noise

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
            for ind in range(len(svm_vec)):
                c = np.random.normal(0, scale=bean_noise[C][ind])
                svm_vec[ind] += c
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
        bean_acc[C] = (avg_orig_acc, avg_priv_acc)
        print(f'acc={avg_orig_acc}, {avg_priv_acc}')

    with open(f'test_data/bean_svm_acc_auto_s=0.5_mi={mi}.pkl', 'wb') as f:
        pickle.dump(bean_acc, f)
        
    with open(f'test_data/bean_svm_noise_auto_s=0.5_mi={mi}.pkl', 'wb') as f:
        pickle.dump(bean_noise, f)
