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

train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True)
subsample_rate = int(0.5*train_len)

regularizations = [(None, 0, 1.0), (0.01, 0.1, 0.51), (0.01, 0.2, 0.51), (0.01, 0.3, 0.51), (0.01, 0.4, 0.51), (0.01, 0.5, 0.51),
                   (0.01, 0.1, 0.6), (0.01, 0.1, 0.7), (0.01, 0.1, 0.8), (0.01, 0.1, 0.9), (0.01, 0.1, 1.0),]
num_trees = 3
tree_depth = 5
mi = 0.5
print("DATA LOADED")

for reg in regularizations:
    print(f'regularize = {reg}, mi = {mi}')

    noise = {}
    est_noise = hybrid_noise_auto(train_x, train_y, fit_forest, subsample_rate, eta=1e-3,
        num_classes = num_classes, max_mi=mi, regularize=reg, num_trees = num_trees, tree_depth=tree_depth)
    noise[reg] = est_noise
    print(f'bean noise {est_noise}')
    with open(f'hybrid_dt/bean_noise_auto_reg={reg}.pkl', 'wb') as f:
        pickle.dump(noise, f)    

