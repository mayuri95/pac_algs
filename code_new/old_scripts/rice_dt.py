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
from algs_lib_copy import *
import sys

train_x, train_y, test_x, test_y, num_classes, train_len = gen_rice(normalize=True)
subsample_rate = int(0.5*train_len)

# (None, 0, 1.0),
regularizations = [(None, 0, 1.0), (0.01, 0.35, 0.51)]
num_trials = 1000
num_trees = 3
tree_depth = 3
print("DATA LOADED")


mi_range = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]

for reg in regularizations:
    for mi in mi_range:
        print(f'regularize = {reg}, mi = {mi}')

        noise = {}
        est_noise = hybrid_noise_auto(train_x, train_y, fit_forest, subsample_rate, eta=1e-6,
            num_classes = num_classes, max_mi=mi, regularize=reg, num_trees = num_trees, tree_depth=tree_depth)
        noise[reg] = est_noise
        print(f'rice noise {est_noise}')

        with open(f'test_data/rice_noise_auto_reg={reg}_mi={mi}.pkl', 'wb') as f:
            pickle.dump(noise, f)    

        num_features = len(train_x[0])
        acc_dict = {}
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
            ordered_feats = get_ordered_feats(num_features, num_trees, tree_depth, None)
            forest, forest_vec = fit_forest(shuffled_x1, shuffled_y1, num_trees, tree_depth, regularize=reg, seed=None)
            acc = forest.calculate_accuracy(test_x, test_y)
            avg_orig_acc += acc
            forest.add_noise_aniso(noise[reg])
            priv_acc = forest.calculate_accuracy(test_x, test_y)
            avg_priv_acc += priv_acc
            
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[reg] = (avg_orig_acc, avg_priv_acc)
        print(f'rice acc = {(avg_orig_acc, avg_priv_acc)}')



        with open(f'test_data/rice_acc_auto_s=0.5_reg={reg}_mi={mi}.pkl', 'wb') as f:
            pickle.dump(acc_dict, f)
