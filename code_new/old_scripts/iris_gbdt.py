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

train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True)
subsample_rate = int(0.5*train_len)

mi_range = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
mi_range = [1.0]
regularizations = [(None, 0, 1.0), (0.01, 0.68, 0.51)]
num_trials = 10
num_trees = 3
tree_depth = 3
num_trials_int = 1 # todo: fix
print("DATA LOADED")


for mi in mi_range:
    acc_dict = {}
    for reg in regularizations:
        print(f'regularize = {reg}, mi = {mi}')
        
        avg_noise = {}
        for _ in range(num_trials_int):
            prev_models = []
            for tree in range(num_trees): # for each tree
                
                if len(prev_models) > 0:
                    residual_y = []
                    for i in range(len(train_x)):
                        pred_y = 0
                        for j in range(len(prev_models)):
                            pred_y += prev_models[j].classify(train_x[i])
                        residual_y.append(train_y[i] - pred_y)
                else:
                    residual_y = train_y # calculate residuals

                est_noise = hybrid_noise_auto(train_x, residual_y, fit_forest, subsample_rate, eta=1e-3,
                    num_classes = num_classes, max_mi=mi, regularize=reg, num_trees = 1, tree_depth=tree_depth) # estimate noise
                if reg not in avg_noise:
                    if tree not in avg_noise:
                        avg_noise[tree] = {}
                    avg_noise[tree][reg] = est_noise
                else:
                    for ind in range(len(avg_noise[reg])):
                        avg_noise[tree][reg][ind] += est_noise[ind]

                shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)

                shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
                num_features = len(train_x[0])
                ordered_feats = get_ordered_feats(num_features, num_trees, tree_depth, None)
                forest, forest_vec = fit_forest(shuffled_x1, shuffled_y1, num_trees, tree_depth, regularize=reg, seed=None)

                forest.add_noise_aniso(avg_noise[tree][reg])
                prev_models.append(forest) # calculate new residuals
        for tree in range(num_trees):
            for reg in avg_noise[tree]:
                for ind in range(len(avg_noise[tree][reg])):
                    avg_noise[tree][reg][ind] /= num_trials_int
            with open(f'test_data/iris_noise_auto_reg={reg}_tree={tree}_mi={mi}.pkl', 'wb') as f:
                pickle.dump(avg_noise, f)
        
        priv_acc = 0.
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
            ordered_feats = get_ordered_feats(num_features, num_trees, tree_depth, None)
            models = []
            res_y = copy.deepcopy(shuffled_y1)
            for tree in range(num_trees):
                forest, forest_vec = fit_forest(shuffled_x1, res_y, 1, tree_depth, regularize=reg, seed=None)
                forest.add_noise_aniso(avg_noise[tree][reg])
                models.append(forest)
                pred_y = []
                new_res_y = []
                for i in range(len(shuffled_x1)):
                    pred_y = 0
                    for j in range(len(models)):
                        pred_y += models[j].classify(shuffled_x1[i])
                    new_res_y.append(shuffled_y1[i] - pred_y)
                res_y = new_res_y

            preds = []
            for i in range(len(test_x)):
                pred = 0
                for tree in range(num_trees):
                    pred += models[tree].classify(test_x[i])
                if pred == test_y[i]:
                    priv_acc += 1
            priv_acc /= len(test_x)

        priv_acc /= num_trials
        acc_dict[reg] = priv_acc
        print(f'iris acc = {priv_acc}')

        with open(f'test_data/iris_acc_auto_reg={reg}_mi={mi}.pkl', 'wb') as f:
            pickle.dump(acc_dict, f)
                                               





