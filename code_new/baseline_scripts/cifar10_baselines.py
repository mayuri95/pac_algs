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

# cifar10

train_x, train_y, test_x, test_y, num_classes, train_len = gen_cifar10(normalize=True)

dims = [1, 3]

num_trials = 1000
subsample_rate = train_len
seed = 743895091

baseline_accs = {}

# PCA

baseline_accs['pca'] = {}
for dim in dims:
    avg_acc = 0
    for i in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = shuffled_x1[:subsample_rate], shuffled_y1[:subsample_rate]
        model, components = run_pca(shuffled_x1, shuffled_y1, num_dims=dim, seed = seed)
        predictions = model.inverse_transform(model.transform(test_x))
        acc = np.linalg.norm(test_x - predictions)
        acc /= np.linalg.norm(test_x)
        avg_acc += acc
    avg_acc /= num_trials
    baseline_accs['pca'][dim] = avg_acc

with open('hybrid_baseline/cifar10_baselines.pkl', 'wb') as f:
	pickle.dump(baseline_accs, f)


