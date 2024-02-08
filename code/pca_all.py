from keras.datasets import mnist
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
import pickle
import keras
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

mi_range = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]


# LOAD IRIS
iris_train_x, iris_train_y, iris_test_x, iris_test_y, iris_num_classes, iris_train_samples = gen_iris()
subsample_rate = int(0.5*iris_train_samples)


test_dims = [1, 2, 3]
num_trials = 1000


# train_x, train_y, mechanism, subsample_rate, num_classes=None, num_dims=None, prefix=None, max_mi = 1.
for mi in mi_range:
    noise = {}

    for num_features in test_dims:
        est_noise = rand_mechanism_noise(iris_train_x, iris_train_y, run_pca, subsample_rate, tau=3,
            num_dims=num_features, num_classes = iris_num_classes, max_mi=mi)[2]
        noise[num_features] = est_noise

    with open(f'data_0120/pca_noise_iris_mi_{mi}.pkl', 'wb') as f:
        pickle.dump(noise, f)


    acc_dict = {}
    for num_features in test_dims:
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(iris_train_x, iris_train_y)
            shuffled_x1, shuffled_y1 = shuffled_x1[:subsample_rate], shuffled_y1[:subsample_rate]
            model, components = run_pca(shuffled_x1, shuffled_y1, num_dims=num_features)
            predictions = model.inverse_transform(model.transform(iris_test_x))

            acc = np.linalg.norm(iris_test_x - predictions)
            acc /= np.linalg.norm(iris_test_x)
            avg_orig_acc += acc
            shape = components.shape
            comp = components.flatten()
            for ind in range(len(comp)):
                c = np.random.normal(0, scale=noise[num_features])
                comp[ind] += c
            new_comp = np.reshape(comp, shape)
            model.components_ = new_comp
            new_predictions = model.inverse_transform(model.transform(iris_test_x))

            priv_acc = np.linalg.norm(iris_test_x - new_predictions)
            priv_acc /= np.linalg.norm(iris_test_x)
            avg_priv_acc += priv_acc
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[num_features] = (avg_orig_acc, avg_priv_acc)

    print(mi, noise, acc_dict)

    with open(f'data_0120/pca_dist_iris_mi_{mi}.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)



# LOAD rice
train_x, train_y, test_x, test_y, num_classes, train_len = gen_rice(normalize=True)
subsample_rate = int(0.5*train_len)

test_dims = list(range(1, 6))

for mi in mi_range:
    noise = {}

    for num_features in test_dims:

        est_noise = rand_mechanism_noise(train_x, train_y, run_pca, subsample_rate, tau=3,
            num_dims=num_features, num_classes = num_classes, max_mi=mi)[2]
        noise[num_features] = est_noise

    with open(f'data_0120/pca_noise_rice_mi_{mi}.pkl', 'wb') as f:
        pickle.dump(noise, f)

    acc_dict = {}
    for num_features in test_dims:
        avg_orig_acc = 0
        avg_priv_acc = 0
        for i in range(num_trials):
            shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
            shuffled_x1, shuffled_y1 = shuffled_x1[:subsample_rate], shuffled_y1[:subsample_rate]
            model, components = run_pca(shuffled_x1, shuffled_y1, num_dims=num_features)
            predictions = model.inverse_transform(model.transform(test_x))

            acc = np.linalg.norm(test_x - predictions)
            acc /= np.linalg.norm(test_x)
            avg_orig_acc += acc
            shape = components.shape
            comp = components.flatten()
            for ind in range(len(comp)):
                c = np.random.normal(0, scale=noise[num_features])
                comp[ind] += c
            new_comp = np.reshape(comp, shape)
            model.components_ = new_comp
            new_predictions = model.inverse_transform(model.transform(test_x))

            priv_acc = np.linalg.norm(test_x - new_predictions)
            priv_acc /= np.linalg.norm(test_x)
            avg_priv_acc += priv_acc
        avg_orig_acc /= num_trials
        avg_priv_acc /= num_trials
        acc_dict[num_features] = (avg_orig_acc, avg_priv_acc)

    print(mi, noise, acc_dict)

    with open(f'data_0120/pca_dist_rice_mi_{mi}.pkl', 'wb') as f:
        pickle.dump(acc_dict, f)
