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


mi = 0.5

test_dims = [1, 6]

for num_dims in test_dims:
    noise = {}
    train_x, train_y, test_x, test_y, num_classes, train_len = gen_rice(normalize=True, norm_kind='minmax')
    subsample_rate = int(0.5*train_len)

    noise[num_dims] = hybrid_noise_auto(train_x, train_y, run_pca, subsample_rate, eta=1e-6, num_dims = num_dims,
        num_classes = num_classes, max_mi=mi, max_num_trials=10)
    # with open(f'noise/rice_pca_noise_auto_dim={num_dims}.pkl', 'wb') as f:
    #     pickle.dump(noise, f)
