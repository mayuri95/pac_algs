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

regularizations = [(0.05, 0.25, 0.65)]
norms = ['standard_no_std']

ind = 0
regularizations = [regularizations[ind]]
norms = [norms[ind]]
num_trees = 3
tree_depth = 5
mi = 0.5
print(norms)


for reg_i, reg in enumerate(regularizations):
    norm = norms[reg_i]
    noise={}
    train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True, norm_kind=norm)
    subsample_rate = int(0.5*train_len)

    print(f'regularize = {reg}, mi = {mi}')

    est_noise = hybrid_noise_test(train_x, train_y, fit_forest, subsample_rate, eta=None,
        num_classes = num_classes, max_mi=mi, regularize=reg, num_trees = num_trees, tree_depth=tree_depth)
    noise[(norm, reg)] = est_noise
    with open(f'noise/bean_dt_noise_reg={reg}_norm={norm}.pkl', 'wb') as f:
        pickle.dump(noise, f)


