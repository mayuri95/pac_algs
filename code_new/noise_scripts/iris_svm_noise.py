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

train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True)

subsample_rate = int(0.5*train_len)

noise = {}
mi = 0.5
# C_range = [x / 100 for x in range(0, 101, 5)][1:]
C_range = [0.05, 1.0]

noise = {}
for C in C_range:
    print(f"C={C}, mi={mi}")
    
    est_x, est_noise = hybrid_noise_test(train_x, train_y, run_svm, subsample_rate, eta=1e-6,
        num_classes = num_classes, max_mi=mi, regularize=C, record_ys = True, fname = f'hybrid_svm/iris_svm_C={C}_ys.pkl')
    noise[C] = (est_noise, est_x)
    
with open(f'data/iris_svm_noise.pkl', 'wb') as f:
    pickle.dump(noise, f)