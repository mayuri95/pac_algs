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

test_vals = list(range(100))

train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True)
C_vals = [1.0, 0.05] # 0.02
with open(f'data/iris_svm_noise.pkl', 'rb') as f:
    all_noise = pickle.load(f) # keyed by MI, default is 0.5

mi_range = [1/128., 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

num_features = len(train_x[0])

subsample_rate = int(0.5*train_len)
num_models = 5000 # num shadow models (256)
seed = 743895091


for add_noise in [True, False]:
    for C in C_vals:
        for trial_ind in test_vals:
            for mi in mi_range:
                false_dists = []
                orig_noise, xs = copy.deepcopy(all_noise[C])
                max_noise = max(orig_noise.values())
                scaled_noise = {k: max_noise * (0.5 / mi) for k in orig_noise}
                print(f'trial number {trial_ind}')
                test_ind = trial_ind

                keys = [k for k in xs if trial_ind not in xs[k]]

                # create false dists
                for model_i in range(num_models):
                    if model_i % 100 == 0:
                        print(f'model {model_i}')

                    choice = np.random.choice(keys)

                    shuffled_x1, shuffled_y1 = train_x[xs[choice]], train_y[xs[choice]]

                    model, svm_vec = run_svm(shuffled_x1, shuffled_y1, num_classes=num_classes, seed=seed,
                                                 regularize=C)
                    if add_noise:

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
                    false_dists.append(conf)

                with open(f'hybrid_lr/models/iris_svm_iso_C={C}_noise={add_noise}_ind_{trial_ind}_mi={mi}_dist.pkl', 'wb') as f:
                    pickle.dump(false_dists, f)

