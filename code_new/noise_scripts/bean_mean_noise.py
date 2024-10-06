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

noise = {}
mi = 0.5

for ind in [True, False]:
    if not ind:
        est_x, est_noise = hybrid_noise_mean(train_x, train_y, subsample_rate, eta=1e-6,
            num_classes = num_classes, max_mi=mi)
    if ind:
        est_x, est_noise = hybrid_noise_mean_ind(train_x, train_y, subsample_rate, eta=1e-6,
            num_classes = num_classes, max_mi=mi)
    noise[mi] = (est_noise, est_x)
    with open(f'data/bean_mean_noise_ind={ind}.pkl', 'wb') as f:
        pickle.dump(noise, f)
print('bean noise complete')