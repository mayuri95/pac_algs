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


regularizations = [(None, 0.0, 1.0), (0.01, 0.2, 0.8)]
norms = ['power', 'robust_no_scale']
num_trees = 1
tree_depth = 3

num_trials = 1000
mi_range = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 1/128.]

for reg_i, reg in enumerate(regs):
    acc_dict = {}
    norm = norms[reg_i]
    train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind=norm)
    subsample_rate = int(0.5*train_len)
    with open(f'noise/iris_dt_noise_reg={reg}_norm={norm}.pkl', 'rb') as f:
        orig_noise, seed = pickle.load(f)[(norm,reg)]
    for mi in mi_range:
        scaled_noise = {k: np.sqrt(orig_noise[k] * (0.5 / mi)) for k in orig_noise}
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
            forest, forest_vec = fit_forest(shuffled_x1, shuffled_y1, num_trees, tree_depth, regularize=reg, seed=seed)
            acc = forest.calculate_accuracy(test_x, test_y)
            avg_orig_acc += acc
            forest.add_noise_aniso(scaled_noise, shuffled_x1, shuffled_y1)
            priv_acc = forest.calculate_accuracy(test_x, test_y)
            avg_priv_acc += priv_acc
            
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[(norm, mi)] = (avg_orig_acc, avg_priv_acc)
        print(f'iris acc (reg={reg}, norm={norm}, aniso, mi={mi}): {(avg_orig_acc, avg_priv_acc)}')

    with open(f'accs/iris_acc_auto_reg={reg}_mi={mi}_norm={norm}.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)

for reg_i, reg in enumerate(regs):
    norm = norms[reg_i]
    acc_dict = {}
    train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind=norm)
    subsample_rate = int(0.5*train_len)
    with open(f'noise/iris_dt_noise_reg={reg}_norm={norm}.pkl', 'rb') as f:
        orig_noise, seed = pickle.load(f)[(norm,reg)]
    for mi in mi_range:
        scaled_noise = {k: np.sqrt(orig_noise[k] * (0.5 / mi)) for k in orig_noise}
        max_noise = max(scaled_noise.values())
        scaled_noise = {k: max_noise for k in scaled_noise}
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
            forest, forest_vec = fit_forest(shuffled_x1, shuffled_y1, num_trees, tree_depth, regularize=reg, seed=seed)
            acc = forest.calculate_accuracy(test_x, test_y)
            avg_orig_acc += acc
            forest.add_noise_aniso(scaled_noise, shuffled_x1, shuffled_y1)
            priv_acc = forest.calculate_accuracy(test_x, test_y)
            avg_priv_acc += priv_acc

        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[(norm, mi)] = (avg_orig_acc, avg_priv_acc)
        print(f'iris acc (reg={reg}, norm={norm}, aniso, mi={mi}): {(avg_orig_acc, avg_priv_acc)}')

    with open(f'accs/iris_iso_acc_auto_reg={reg}_mi={mi}_norm={norm}.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)
