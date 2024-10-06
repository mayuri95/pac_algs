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

max_val = int(sys.argv[1])
test_vals = list(range(max_val-20, max_val))

train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True)
C_range = [1.0, 0.05] # 0.02
noise_dict = {
    0.05: {0: 0.015065604910835126, 1: 0.030515233065788138, 2: 0.014200608406282626, 3: 0.016315324539198463, 4: 0.03707567668257179,
          5: 0.03495278778301176, 6: 0.03296876082166603, 7: 0.03186689655881539,
          8: 0.03421765502677221, 9: 0.043902084301947135, 10: 0.03135299962641335, 11: 0.02905825409135171,
          12: 0.02822000068661683, 13: 0.029649917352618856, 14: 0.02522317870040466},
    0.05: {0: 0.2134861159877339, 1: 0.36671982224891275, 2: 0.1887792907484354, 3: 0.19933317940941472,
            4: 0.28907624575344243, 5: 1.281864432818346, 6: 1.0227687564208732, 7: 0.8625464757825708, 8: 0.9087228467675248,
            9: 0.697516150069243, 10: 0.9410082448605513, 11: 0.9926843379790549, 12: 0.6055792734592831, 13: 0.6480364080636882,
            14: 0.43930527890838045}
    }

mi_range = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

num_features = len(train_x[0])

subsample_rate = int(0.5*train_len)
num_models = 1000 # num shadow models (256)

for add_noise in [True, False]:
    for C in C_vals:
        for trial_ind in test_vals:
            for mi in mi_range:
                false_dists = []
                orig_noise = copy.deepcopy(noise_dict[C])
                max_noise = max(orig_noise.values())
                scaled_noise = {k: max_noise * (0.5 / mi) for k in orig_noise}
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

                    shuffled_x1, shuffled_y1 = shuffle(other_x, other_y)

                    shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
                    
                    model, svm_vec = run_svm(shuffled_x1, shuffled_y1, num_classes=num_classes, seed=None,
                                                 regularize=C)
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

                with open(f'hybrid_data/test_lr_noise/models/iris_iso_svm_C={C}_noise={add_noise}_ind_{trial_ind}_mi={mi}_dist.pkl', 'wb') as f:
                    pickle.dump(false_dists, f)
