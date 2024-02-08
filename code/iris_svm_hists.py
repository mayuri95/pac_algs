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
test_vals = list(range(max_val-10, max_val))

train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True)
num_features = len(train_x[0])

subsample_rate = int(0.5*train_len)

C_range = [1.0, 0.02] # 0.02

noise_vals = {
    1.0: {
        4.0: 0.330613386180837,
        1.0: 1.3140991428547941,
        0.25: 6.095456013318111,
        0.0625: 22.94876523327606,
        0.015625: 93.3509039071466},
    0.02: {
        4.0: 0.0069028968315583,
        1.0: 0.02755094686548948,
        0.25: 0.1220526168705784,
        0.0625: 0.47054463112107864,
        0.015625: 1.8719401974447367}
    }

num_models = 5000 # num shadow models (256)

for C_ind, C in enumerate(C_range):
    print(f'C={C}')
    for trial_ind in test_vals:
        for mi in noise_vals[C]:
            false_dists = []
            print(f'trial number {trial_ind}')
            test_ind = trial_ind

            # create false dists
            for _ in range(num_models):
                other_x = np.delete(train_x, test_ind, 0)
                other_y = np.delete(train_y, test_ind, 0)
                other_x = copy.deepcopy(other_x)
                other_y = copy.deepcopy(other_y)
                shuffled_x1, shuffled_y1 = shuffle(other_x, other_y)
                shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
                model, svm_vec = run_svm(shuffled_x1, shuffled_y1, num_classes=num_classes, seed=None,
                                             regularize=C)
                for ind in range(len(svm_vec)):
                    c = np.random.normal(0, scale=noise_vals[C][mi])
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
            with open(f'test_lr_noise/models/iris_svm_ind_C={C}_{trial_ind}_mi={mi}_noised_dist.pkl', 'wb') as f:
                pickle.dump(false_dists, f) 
