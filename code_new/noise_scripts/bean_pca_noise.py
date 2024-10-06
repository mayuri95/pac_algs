from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
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
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from algs_lib import *
import sys


# LOAD bean
train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True)
subsample_rate = int(0.5*train_len)

mi = 0.5
#test_dims = list(range(1, len(train_x[0])))
test_dims = [1, 8]

for num_dims in test_dims:
    est_x, est_noise = hybrid_noise_test(train_x, train_y, run_pca, subsample_rate, eta=1e-6, num_dims = num_dims,
        num_classes = num_classes, max_mi=mi)
    with open(f'data/bean_pca_noise_auto_dim={num_dims}.pkl', 'wb') as f:
        pickle.dump((est_noise, est_x), f)
