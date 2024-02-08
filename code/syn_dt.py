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

big = eval(sys.argv[1])
if big:
    mi_range = [0.5]
else:
    mi_range = [0.03125, 0.015625]
print(f"BIG={big}")


train_x, train_y, test_x, test_y, num_classes, train_len = gen_synthetic(normalize=True, num_train=10000, num_test=3000)
subsample_rate = int(0.5*train_len)


regularizations = [(None, 0, 1.0)]
num_trials = 1000
num_trees = 1
tree_depth = 3
print("DATA LOADED")


for mi in mi_range:

    for reg in regularizations:
        print(f'regularize = {reg}, mi = {mi}')

        noise = {}
        est_noise =  rand_mechanism_noise(train_x, train_y, fit_forest, subsample_rate, num_classes = num_classes,
                                     num_trees = num_trees, tree_depth=tree_depth, regularize=reg, tau=3,
                                     prefix='data_1129/syn_')[2]
        noise[reg] = est_noise
        print(f'syn noise {est_noise}')

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
            forest.add_noise(noise[reg])
            priv_acc = forest.calculate_accuracy(test_x, test_y)
            avg_priv_acc += priv_acc
            
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[reg] = (avg_orig_acc, avg_priv_acc)
        print(f'syn acc = {(avg_orig_acc, avg_priv_acc)}')

        with open(f'data_0120/syn_noise_reg={reg}_mi={mi}.pkl', 'wb') as f:
            pickle.dump(noise, f)    

        with open(f'data_0120/syn_acc_reg={reg}_mi={mi}.pkl', 'wb') as f:
            pickle.dump(acc_dict, f)
