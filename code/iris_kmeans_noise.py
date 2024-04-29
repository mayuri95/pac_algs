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

est_noise = hybrid_noise_auto(train_x, train_y, run_kmeans, subsample_rate, eta=1e-6,
    num_classes = num_classes, max_mi=mi)
noise[mi] = est_noise
with open(f'hybrid_data/iris_kmeans_hybrid_auto_s=0.5_noise.pkl', 'wb') as f:
    pickle.dump(noise, f)
print('iris noise complete')