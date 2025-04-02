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


norms = ['power_standard', 'robust']
C_vals = [1e-6, 1.0]
with open(f'noise/iris_svm_noise.pkl', 'rb') as f:
    all_noise = pickle.load(f) # keyed by MI, default is 0.5

mi_range = [1/128., 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]


mi_ind = int(sys.argv[1])
mi_range = mi_range[mi_ind:mi_ind+2]
print(mi_range)

num_models = 1000 # num shadow models (256)
seed = 743895091

for iso in [True, False]:
    for add_noise in [True, False]:
        for C_ind, C in enumerate(C_vals):
            norm = norms[C_ind]
            train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind=norm)

            num_features = len(train_x[0])

            subsample_rate = int(0.5*train_len)
            for trial_ind in test_vals:
                for mi in mi_range:
                    if add_noise is False and mi != 1/128:
                        continue
                    false_dists = []
                    orig_noise, seed = copy.deepcopy(all_noise[(C, norm)])
                    scaled_noise = {k: np.sqrt(orig_noise[k] * (0.5 / mi)) for k in orig_noise}
                    if iso:
                        max_noise = max(scaled_noise.values())
                        scaled_noise = {k: max_noise for k in scaled_noise}
                    print(f'trial number {trial_ind}')
                    test_ind = trial_ind

                    # create false dists
                    for model_i in range(num_models):
                        if model_i % 100 == 0:
                            print(f'model {model_i}')

                        other_x = np.delete(train_x, test_ind, 0)
                        # print(other_x)
                        other_y = np.delete(train_y, test_ind, 0)

                        other_x = copy.deepcopy(other_x)
                        other_y = copy.deepcopy(other_y)

                        shuffled_x, shuffled_y= shuffle(other_x, other_y)

                        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate)
                        
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


                    if iso:
                        isof = '_iso'
                    else:
                        isof = ''
                    with open(f'hybrid_lr/models/iris{isof}_svm_C={C}_noise={add_noise}_ind_{trial_ind}_mi={mi}_dist.pkl', 'wb') as f:
                        pickle.dump(false_dists, f)

