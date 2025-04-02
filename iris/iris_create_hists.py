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

test_vals = list(range(100))

mi_range = [1/128., 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

mi_ind = int(sys.argv[1])
mi_range = mi_range[mi_ind:mi_ind+2]
print(mi_range)

num_models = 1000 # num shadow models (256)
seed = 743895091

for iso in [True, False]:
    for rebalance in [False]:
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind = 'power')
        with open(f'noise/iris_kmeans_hybrid_auto_s=0.5_noise_rebalance=False.pkl', 'rb') as f:
            orig_noise, seed = pickle.load(f)[0.5] # keyed by MI, default is 0.5
        num_features = len(train_x[0])

        subsample_rate = int(0.5*train_len)
        for add_noise in [True, False]:
            for trial_ind in test_vals:
                for mi in mi_range:
                    if add_noise is False and mi != 1/128:
                        continue
                    false_dists = []

                    scaled_noise = {k: np.sqrt(orig_noise[k] * (0.5 / mi)) for k in orig_noise}
                    if iso:
                        max_noise = max(scaled_noise.values())
                        scaled_noise = {k: max_noise for k in scaled_noise}
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

                        shuffled_x, shuffled_y= shuffle(other_x, other_y)

                        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate)
                        model, cluster_centers = run_kmeans(shuffled_x1, shuffled_y1, num_clusters=num_classes, seed=seed, rebalance=rebalance)

                        cluster_centers = np.array(cluster_centers).reshape((num_classes, -1))
                        # add noise
                        if add_noise:
                            for ind in range(len(cluster_centers)):
                                c = np.random.normal(0, scale=scaled_noise[ind])
                                cluster_centers[ind] += c
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

                    if iso:
                        isof = '_iso'
                    else:
                        isof = ''
                    with open(f'hybrid_lr/models/iris{isof}_kmeans_noise={add_noise}_ind_{trial_ind}_mi={mi}_dist.pkl', 'wb') as f:
                        pickle.dump(false_dists, f)