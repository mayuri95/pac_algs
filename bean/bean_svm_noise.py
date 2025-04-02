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

norms = ['quantile_gaussian', 'robust']
C_range = [0.001, 1.0]

noise={}
for norm_ind, norm in enumerate(norms):
    train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True, norm_kind=norm)

    subsample_rate = int(0.5*train_len)

    mi = 0.5
    C = C_range[norm_ind]

    print(f"C={C}, mi={mi}")

    est_noise = hybrid_noise_auto(train_x, train_y, run_svm, subsample_rate, eta=1e-6,
        num_classes = num_classes, max_mi=mi, regularize=C)
    noise[(C, norm)] = est_noise
for (C, norm) in noise:
    print(norm, C, np.linalg.norm(list(noise[(C,norm)][0].values())))
with open(f'noise/bean_svm_noise.pkl', 'wb') as f:
    pickle.dump(noise, f)