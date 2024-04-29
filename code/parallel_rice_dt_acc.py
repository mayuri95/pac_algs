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
mi_range = [mi]

train_x, train_y, test_x, test_y, num_classes, train_len = gen_rice(normalize=True)
subsample_rate = int(0.5*train_len)

regs = [(None, 0, 1.0), (0.01, 0.35, 0.51)]
num_trees = 3
tree_depth = 3


num_trials = 1000

for reg in regs:
    with open(f'hybrid_data/rice_noise_auto_reg={reg}.pkl', 'rb') as f:
        orig_noise = pickle.load(f)[reg]
    for mi in mi_range:
        acc_dict = {}
        scaled_noise = {k: orig_noise[k] * (0.5 / mi) for k in orig_noise}
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
            forest, forest_vec = fit_forest(shuffled_x1, shuffled_y1, num_trees, tree_depth, regularize=reg, seed=None)
            acc = forest.calculate_accuracy(test_x, test_y)
            avg_orig_acc += acc
            forest.add_noise_aniso(scaled_noise)
            priv_acc = forest.calculate_accuracy(test_x, test_y)
            avg_priv_acc += priv_acc
            
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[reg] = (avg_orig_acc, avg_priv_acc)
        print(f'rice acc = {(avg_orig_acc, avg_priv_acc)}')

        with open(f'hybrid_data/rice_acc_auto_reg={reg}_mi={mi}.pkl', 'wb') as f:
            pickle.dump(acc_dict, f)

for reg in regs:
    with open(f'hybrid_data/rice_noise_auto_reg={reg}.pkl', 'rb') as f:
        orig_noise = pickle.load(f)[reg]
    for mi in mi_range:
        acc_dict = {}
        scaled_noise = {k: orig_noise[k] * (0.5 / mi) for k in orig_noise}
        max_noise = max(scaled_noise.values())
        scaled_noise = {k: max_noise for k in scaled_noise}
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
            forest, forest_vec = fit_forest(shuffled_x1, shuffled_y1, num_trees, tree_depth, regularize=reg, seed=None)
            acc = forest.calculate_accuracy(test_x, test_y)
            avg_orig_acc += acc
            forest.add_noise_aniso(scaled_noise)
            priv_acc = forest.calculate_accuracy(test_x, test_y)
            avg_priv_acc += priv_acc
            
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[reg] = (avg_orig_acc, avg_priv_acc)
        print(f'rice acc = {(avg_orig_acc, avg_priv_acc)}')

        with open(f'hybrid_data/rice_iso_acc_auto_reg={reg}_mi={mi}.pkl', 'wb') as f:
            pickle.dump(acc_dict, f)
