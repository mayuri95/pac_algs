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

test_vals = list(range(0, 100))

train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True)

num_features = len(train_x[0])

subsample_rate = int(0.5*train_len)
num_models = 1000 # num shadow models (256)
seed = 743895091

for add_noise in [False]:
    for trial_ind in test_vals:
            false_dists = []
            print(f'trial number {trial_ind}')
            test_ind = trial_ind

            # create false dists
            for model_i in range(num_models):
                if model_i % 100 == 0:
                    print(f'model {model_i}')
                other_x = np.delete(train_x, test_ind, 0)
                # print(other_x)
                other_y = np.delete(train_y, test_ind, 0)

                other_x = copy.deepcopy(other_x)
                other_y = copy.deepcopy(other_y)

                shuffled_x1, shuffled_y1 = shuffle(other_x, other_y)

                shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)

                model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=seed, rebalance=False)

                cluster_centers = np.array(cluster_centers).reshape((num_classes, -1))
                new_centers = np.reshape(cluster_centers, (num_classes, num_features))
                model.cluster_centers_ = new_centers

                cluster_id = model.predict([train_x[test_ind]])[0]

                all_dists = []
                num_clusters = len(cluster_centers)
                for cid in range(num_clusters):

                    all_dists.append(np.linalg.norm(train_x[test_ind] - cluster_centers[cid]))
                norm_factor = sum(all_dists)
                all_dists = [k/norm_factor for k in all_dists]
                conf = 1-min(all_dists)
                false_dists.append(conf)

            with open(f'hybrid_lr/models/iris_kmeans_noise={add_noise}_ind_{trial_ind}_dist.pkl', 'wb') as f:
                pickle.dump(false_dists, f)

