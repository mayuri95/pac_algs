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

ind = int(sys.argv[1])

num_trees = 1 
tree_depth = 3
mi = 0.5
print(f"DATA LOADED; IND {ind}")
norms = [norms[ind]]
regularizations = [regularizations[ind]]
for i, reg in enumerate(regularizations):
        norm = norms[i]
        print(norm, reg)
        noise={}
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind=norm)
        subsample_rate = int(0.5*train_len)


        est_noise = hybrid_noise_auto(train_x, train_y, fit_forest, subsample_rate, eta=1e-6,
            num_classes = num_classes, max_mi=mi, regularize=reg, num_trees = num_trees, tree_depth=tree_depth)
        noise[(norm, reg)] = est_noise
        with open(f'noise/iris_dt_noise_reg={reg}_norm={norm}.pkl', 'wb') as f:
            pickle.dump(noise, f)


