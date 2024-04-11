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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo 
from collections import Counter
from scipy.stats import entropy
from sklearn import tree
from sklearn import preprocessing
from itertools import product

# GET SAMPLES PER CLASS
def get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate, ordered=False):
    init_seed_samples = []
    init_y = []
    seen = set()
    seeded_x = copy.deepcopy(shuffled_x)
    seeded_y = copy.deepcopy(shuffled_y)
    remaining = subsample_rate - num_classes

    assert remaining > 0

    to_delete = []
    for ind, x in enumerate(shuffled_x):
        if ordered:
            classification = shuffled_y[ind][0]
        else:
            classification = shuffled_y[ind]
        if classification not in seen and len(seen) < num_classes:
            init_seed_samples.append(x)
            seen.add(classification)
            init_y.append(shuffled_y[ind])
            to_delete.append(ind)
    to_delete.reverse()
    for ind in to_delete:
        seeded_x = np.delete(seeded_x, ind, 0)
        seeded_y = np.delete(seeded_y, ind, 0)

    init_seed_samples = np.array(init_seed_samples)

    all_samples = np.vstack((init_seed_samples, seeded_x[:remaining]))
    if ordered:
        all_y = np.vstack((init_y, seeded_y[:remaining]))
    else:
        all_y = np.hstack((init_y, seeded_y[:remaining]))
    return all_samples, all_y

# NOISE MECHANISMS

def calc_r(train_x):
    max_l2_norm = None
    for x in train_x:
        if max_l2_norm is None or np.linalg.norm(x) > max_l2_norm:
            max_l2_norm = np.linalg.norm(x)
    return max_l2_norm

def rand_mechanism_noise(train_x, train_y, mechanism, subsample_rate, tau, num_classes, regularize=False, tree_depth=None,
    num_trees=None, num_dims=None, prefix=None, max_mi = 1., sec_c = 1e-6):
    v = max_mi / 2.
    r = calc_r(train_x)
    gamma = 0.01
    c = sec_c
    num_trials = 1000
    avg_dist = 0.
    
    ys = []
    train_y = [(train_y[k], k) for k in range(len(train_y))]

    for trial in range(num_trials):
        print(f'trial {trial}')
        assert subsample_rate >= num_classes
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        print(shuffled_x1[0])
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate, ordered=True)

        indices = [k[1] for k in shuffled_y1.argsort(axis=0)]

        shuffled_x1 = shuffled_x1[indices]
        shuffled_y1 = shuffled_y1[indices]
        
        shuffled_y1 = np.array([k[0] for k in shuffled_y1])

        shuffled_x2, shuffled_y2 = shuffle(train_x, train_y)
        shuffled_x2, shuffled_y2 = get_samples_safe(shuffled_x2, shuffled_y2, num_classes, subsample_rate, ordered=True)

        indices = [k[1] for k in shuffled_y2.argsort(axis=0)]

        shuffled_x2 = shuffled_x2[indices]
        shuffled_y2 = shuffled_y2[indices]
        shuffled_y2 = np.array([k[0] for k in shuffled_y2])
        
        seeds = list(range(tau))
        
        
        if mechanism == run_kmeans:
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_classes, seeds[i])[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_classes, seeds[i])[1] for i in range(len(seeds))]
        
        if mechanism == run_svm:
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_classes, seeds[i], regularize)[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_classes, seeds[i], regularize)[1] for i in range(len(seeds))]

        if mechanism.__name__ == 'fit_forest' or mechanism.__name__ == 'fit_gbdt':
            assert num_trees is not None
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_trees, tree_depth, seeds[i], regularize)[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_trees, tree_depth, seeds[i], regularize)[1] for i in range(len(seeds))]

        if mechanism.__name__ == 'run_pca':
            assert num_dims is not None
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_dims)[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_dims)[1] for i in range(len(seeds))]
            ordered_y1 = []
            ordered_y2 = []
            for ind in range(len(y_1)):
                dist = 0.
                s1 = y_1[ind]
                s2 = y_2[ind]
                ordered_y1.append(s1.flatten())
                    

                s_1 = copy.deepcopy(s1)
                s_2 = copy.deepcopy(s2)

                u_a, s_a, v_a = np.linalg.svd(s_1)
                u_b, s_b, v_b = np.linalg.svd(s_2)

                c_mat = np.matmul(v_a, np.transpose(v_b))

                end_shape = s2.shape[0]
                c_trunc = c_mat[:end_shape, :end_shape]
                transformed_s2 = np.matmul(c_trunc, s2)

                for i in range(len(s2)):
                    orig_dist = np.linalg.norm(s2[i] - s1[i])
                    neg_dist = np.linalg.norm(-1*s2[i] - s1[i])
                    if neg_dist < orig_dist:
                        s2[i] *= -1
                s2 = s2.flatten()
                ordered_y2.append(s2)
            y_1 = copy.deepcopy(ordered_y1)
            y_2 = copy.deepcopy(ordered_y2)


        elif mechanism != fit_forest and mechanism != fit_gbdt and mechanism != run_pca:
            # solve LP for optimal assignments
            C = distance.cdist(y_1, y_2)
            assignment = linear_sum_assignment(C)
            # sort by order of first cluster, so those values remain stable
            arr = np.dstack(assignment)[0]
            arr = arr[arr[:, 0].argsort()]
            sorted_ids = arr[:, 1]
            y_2 = np.array([y_2[sorted_ids[j]] for j in range(len(sorted_ids))])

        dist = 0.

        for ind in range(tau):
            dist += np.linalg.norm(np.array(y_1[ind]) - np.array(y_2[ind]))**2 / tau
        avg_dist += dist
        ys.append((y_1, y_2))

    # fname = f'{prefix}{mechanism.__name__}_rate={subsample_rate}_regularize={regularize}.pkl'
    # with open(fname, 'wb') as f:
    #     pickle.dump(ys, f)
    
    avg_dist /= num_trials

    avg_cov_norm = (avg_dist + c) / (2*v)

    return (avg_dist, c, avg_cov_norm)

def rand_mechanism_noise_auto(train_x, train_y, mechanism, subsample_rate, tau, num_classes, eta, regularize=False, tree_depth=None,
    num_trees=None, num_dims=None, prefix=None, max_mi = 1., sec_c = 1e-6):
    v = max_mi / 2.
    r = calc_r(train_x)
    gamma = 0.01
    c = sec_c
    avg_dist = 0.
    converged = False
    current_est = None
    
    ys = []
    train_y = [(train_y[k], k) for k in range(len(train_y))]
    curr_trial = 0

    while not converged:

        print(f'trial {curr_trial}')
        assert subsample_rate >= num_classes
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate, ordered=True)

        indices = [k[1] for k in shuffled_y1.argsort(axis=0)]

        shuffled_x1 = shuffled_x1[indices]
        shuffled_y1 = shuffled_y1[indices]
        
        shuffled_y1 = np.array([k[0] for k in shuffled_y1])

        shuffled_x2, shuffled_y2 = shuffle(train_x, train_y)
        shuffled_x2, shuffled_y2 = get_samples_safe(shuffled_x2, shuffled_y2, num_classes, subsample_rate, ordered=True)

        indices = [k[1] for k in shuffled_y2.argsort(axis=0)]

        shuffled_x2 = shuffled_x2[indices]
        shuffled_y2 = shuffled_y2[indices]
        shuffled_y2 = np.array([k[0] for k in shuffled_y2])
        
        seeds = list(range(tau))
        
        
        if mechanism == run_kmeans:
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_classes, seeds[i])[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_classes, seeds[i])[1] for i in range(len(seeds))]
        
        if mechanism == run_svm:
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_classes, seeds[i], regularize)[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_classes, seeds[i], regularize)[1] for i in range(len(seeds))]

        if mechanism.__name__ == 'fit_forest' or mechanism.__name__ == 'fit_gbdt':
            assert num_trees is not None
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_trees, tree_depth, seeds[i], regularize)[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_trees, tree_depth, seeds[i], regularize)[1] for i in range(len(seeds))]

        if mechanism.__name__ == 'run_pca':
            assert num_dims is not None
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_dims)[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_dims)[1] for i in range(len(seeds))]
            ordered_y1 = []
            ordered_y2 = []
            for ind in range(len(y_1)):
                dist = 0.
                s1 = y_1[ind]
                s2 = y_2[ind]
                ordered_y1.append(s1.flatten())
                    

                s_1 = copy.deepcopy(s1)
                s_2 = copy.deepcopy(s2)

                u_a, s_a, v_a = np.linalg.svd(s_1)
                u_b, s_b, v_b = np.linalg.svd(s_2)

                c_mat = np.matmul(v_a, np.transpose(v_b))

                end_shape = s2.shape[0]
                c_trunc = c_mat[:end_shape, :end_shape]
                transformed_s2 = np.matmul(c_trunc, s2)

                for i in range(len(s2)):
                    orig_dist = np.linalg.norm(s2[i] - s1[i])
                    neg_dist = np.linalg.norm(-1*s2[i] - s1[i])
                    if neg_dist < orig_dist:
                        s2[i] *= -1
                s2 = s2.flatten()
                ordered_y2.append(s2)
            y_1 = copy.deepcopy(ordered_y1)
            y_2 = copy.deepcopy(ordered_y2)


        elif mechanism != fit_forest and mechanism != fit_gbdt and mechanism != run_pca:
            # solve LP for optimal assignments
            C = distance.cdist(y_1, y_2)
            assignment = linear_sum_assignment(C)
            # sort by order of first cluster, so those values remain stable
            arr = np.dstack(assignment)[0]
            arr = arr[arr[:, 0].argsort()]
            sorted_ids = arr[:, 1]
            y_2 = np.array([y_2[sorted_ids[j]] for j in range(len(sorted_ids))])


        dist = 0.

        for ind in range(tau):
            dist += np.linalg.norm(np.array(y_1[ind]) - np.array(y_2[ind]))**2 / tau
        avg_dist += dist

        if curr_trial % 10 == 0:        
            if current_est is None:
                current_est = avg_dist / curr_trial
            else:
                print(f'else reached with {abs(avg_dist - current_est)}, {eta}')
                if abs(avg_dist / curr_trial - current_est) < eta:
                    converged = True
                current_est = avg_dist / curr_trial
        curr_trial += 1
        # print(curr_trial, abs(current_est - avg_dist))
    
    # avg_dist /= curr_trial

    avg_cov_norm = (avg_dist + c) / (2*v)
    

    return (avg_dist, c, avg_cov_norm)

def rand_mechanism_individual_noise(train_x, train_y, mechanism, subsample_rate, tau, num_classes, index, regularize=False, tree_depth=None,
    num_trees=None, prefix=None, max_mi = 1.):
    v = max_mi / 2.
    r = calc_r(train_x)
    gamma = 0.01
    c = 0.001
    num_trials = 1000
    avg_dist = 0.
    
    ys = []
    train_y = [(train_y[k], k) for k in range(len(train_y))]

    test_pt_x = train_x[index]
    test_pt_y = train_y[index]

    train_x = np.delete(train_x, index, 0)
    train_y = np.delete(train_y, index, 0)

    for trial in range(num_trials):
        assert subsample_rate >= num_classes
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate, ordered=True)

        indices = [k[1] for k in shuffled_y1.argsort(axis=0)]

        shuffled_x1 = shuffled_x1[indices]
        shuffled_y1 = shuffled_y1[indices]

        shuffled_x2 = copy.deepcopy(shuffled_x1)
        shuffled_y2 = copy.deepcopy(shuffled_y1)
        shuffled_x2[0] = test_pt_x
        shuffled_y2[0] = test_pt_y


        shuffled_y1 = np.array([k[0] for k in shuffled_y1])
        indices = [k[1] for k in shuffled_y2.argsort(axis=0)]

        shuffled_x2 = shuffled_x2[indices]
        shuffled_y2 = shuffled_y2[indices]
        shuffled_y2 = np.array([k[0] for k in shuffled_y2])
        
        seeds = list(range(tau))
        
        
        if mechanism == run_kmeans:
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_classes, seeds[i])[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_classes, seeds[i])[1] for i in range(len(seeds))]
        
        if mechanism == run_svm:
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_classes, seeds[i], regularize)[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_classes, seeds[i], regularize)[1] for i in range(len(seeds))]

        if mechanism.__name__ == 'fit_forest' or mechanism.__name__ == 'fit_gbdt':
            assert num_trees is not None
            y_1 = [mechanism(shuffled_x1, shuffled_y1, num_trees, tree_depth, seeds[i], regularize)[1] for i in range(len(seeds))]
            y_2 = [mechanism(shuffled_x2, shuffled_y2, num_trees, tree_depth, seeds[i], regularize)[1] for i in range(len(seeds))]
        
        if mechanism != fit_forest and mechanism != fit_gbdt:
            # solve LP for optimal assignments
            C = distance.cdist(y_1, y_2)
            assignment = linear_sum_assignment(C)
            # sort by order of first cluster, so those values remain stable
            arr = np.dstack(assignment)[0]
            arr = arr[arr[:, 0].argsort()]
            sorted_ids = arr[:, 1]
            y_2 = np.array([y_2[sorted_ids[j]] for j in range(len(sorted_ids))])

        dist = 0.

        for ind in range(tau):
            dist += np.linalg.norm(np.array(y_1[ind]) - np.array(y_2[ind]))**2 / tau
        avg_dist += dist
        ys.append((y_1, y_2))
    
    avg_dist /= num_trials

    avg_cov_norm = (avg_dist + c) / (2*v)

    return (avg_dist, c, avg_cov_norm)

def calc_cov_large_gap(d, c, v, beta, eigs, u):
    sigma_matrix = np.zeros((d, d))
    for i in range(d):
        num = 2.*v
        denom_init = (eigs[i] + 10*c*v / beta)**0.5
        denom_second = 0.
        for k in range(d):
            denom_second += (eigs[k] + 10*c*v/beta)**0.5
            denom = denom_init * denom_second
            sigma_matrix[i][i] = num/denom
    noise_matrix =  np.matmul(
        np.matmul(u, np.linalg.inv(sigma_matrix)), u.T)
    return noise_matrix

def calc_cov_small_gap(d, c, v, eigs):
    identity = np.identity(d)
    multiplier = sum(eigs) + d*c
    multiplier /= (2*v)
    return multiplier*identity

def hybrid_noise_auto(train_x, train_y, mechanism, subsample_rate, num_classes,
    eta, regularize=None, num_trees=None, tree_depth = None, max_mi = 1.):

    sec_v = max_mi / 2
    sec_beta = max_mi - sec_v
    r = calc_r(train_x)
    gamma = 0.01
    avg_dist = 0.
    curr_est = None
    converged = False
    curr_trial = 0

    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes

    est_y = {}
    prev_ests = None
    # 10*c*v
    seed = np.random.randint(1, 100000)

    while not converged:
        shuffled_x, shuffled_y = shuffle(train_x, train_y)


        shuffled_x, shuffled_y = get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate)

        if mechanism == run_kmeans:
            output = mechanism(shuffled_x, shuffled_y, num_classes, seed)[1]

        if mechanism == run_svm:
            output = mechanism(shuffled_x, shuffled_y, num_classes, seed, regularize)[1]

        if mechanism.__name__ == 'fit_forest' or mechanism.__name__ == 'fit_gbdt':
            assert num_trees is not None
            assert tree_depth is not None
            output = mechanism(shuffled_x, shuffled_y, num_trees, tree_depth, seed, regularize)[1]


        for ind in range(len(output)):
            if ind not in est_y:
                est_y[ind] = []
            est_y[ind].append(output[ind])

        if curr_trial % 10 == 0:        
            if prev_ests is None:
                prev_ests = {}
                for ind in est_y:
                    prev_ests[ind] = np.var(est_y[ind])
            else:
                converged = True
                for ind in est_y:
                    if abs(np.var(est_y[ind]) - prev_ests[ind]) > eta:
                        converged = False
                if not converged:
                    for ind in est_y:
                        prev_ests[ind] = np.var(est_y[ind])
        curr_trial += 1
    fin_var = {ind: np.var(est_y[ind]) for ind in est_y}

    noise = {}
    sqrt_total_var = sum(fin_var.values())**0.5
    for ind in fin_var:
        noise[ind] = 1./(2*max_mi) * fin_var[ind]**0.5 * sqrt_total_var
    return noise

def det_mechanism_noise_auto(train_x, train_y, mechanism, subsample_rate, num_classes,
    eta, regularize=None, max_mi = 1.):

    sec_v = max_mi / 2
    sec_beta = max_mi - sec_v
    r = calc_r(train_x)
    gamma = 0.01
    c = 1e-20
    num_trials = 1000
    avg_dist = 0.
    current_est = None

    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes
    est_y = []
    

    seed = np.random.randint(1, 100000)

    for curr_trial in range(num_trials):
        shuffled_x, shuffled_y = shuffle(train_x, train_y)
        
        
        shuffled_x, shuffled_y = get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate)

        est_y.append(mechanism(shuffled_x, shuffled_y, num_classes, seed, regularize)[1])
        if curr_trial % 10 == 0:        
            if current_est is None:
                current_est = np.linalg.norm(np.cov(np.array(est_y).T)) / curr_trial
            else:
                print(f'else reached with {abs(avg_dist - current_est)}, {eta}')
                avg_cov = np.linalg.norm(np.cov(np.array(est_y).T))
                if abs(avg_cov / curr_trial - current_est) < eta:
                    converged = True
                current_est = avg_cov / curr_trial

    # run_svm(train_x, train_y, num_classes, seed, regularize)


    d = len(est_y[0]) # length of flattened vector
    y_mean = np.average(est_y, axis=0)
    y_cov = np.cov(np.array(est_y).T)  
    
    u, eigs, v = np.linalg.svd(y_cov)
    
    j_0 = 0
    while j_0 < len(eigs) and eigs[j_0] > c:
        j_0 += 1
    if j_0 == len(eigs):
        j_0 -= 1
    elif eigs[j_0] <= c and j_0 > 0:
        j_0 -= 1
    print(f'j0 is {j_0}')
    # calculate eig_gap
    min_eig_gap = None
    for j in range(j_0 + 1):
        for ind in range(len(eigs)):
            if j == ind:
                continue
            eig_gap = abs(eigs[j] - eigs[ind])
            if min_eig_gap is None or eig_gap < min_eig_gap:
                min_eig_gap = eig_gap
    thresh = r * (d*c)**0.5 + 2*c

    print(min_eig_gap, thresh)
    # large eigenvalue gap
    if min_eig_gap > thresh:
        print('anisotropic reached')
        cov = calc_cov_large_gap(d, c, sec_v, sec_beta, eigs, u)
    else:
        cov = calc_cov_small_gap(d, c, sec_v, eigs)
    return cov


def det_mechanism_noise(train_x, train_y, mechanism, subsample_rate, num_classes, regularize=None, max_mi = 1.):

    sec_v = max_mi / 2
    sec_beta = max_mi - sec_v
    r = calc_r(train_x)
    gamma = 0.01
    c = 1e-20
    num_trials = 1000
    avg_dist = 0.

    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes
    est_y = []
    

    seed = np.random.randint(1, 100000)

    for _ in range(num_trials):
        shuffled_x, shuffled_y = shuffle(train_x, train_y)
        
        
        shuffled_x, shuffled_y = get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate)

        est_y.append(mechanism(shuffled_x, shuffled_y, num_classes, seed, regularize)[1])

    # run_svm(train_x, train_y, num_classes, seed, regularize)


    d = len(est_y[0]) # length of flattened vector
    y_mean = np.average(est_y, axis=0)
    y_cov = np.cov(np.array(est_y).T)  
    
    u, eigs, v = np.linalg.svd(y_cov)
    
    j_0 = 0
    while j_0 < len(eigs) and eigs[j_0] > c:
        j_0 += 1
    if j_0 == len(eigs):
        j_0 -= 1
    elif eigs[j_0] <= c and j_0 > 0:
        j_0 -= 1
    print(f'j0 is {j_0}')
    # calculate eig_gap
    min_eig_gap = None
    for j in range(j_0 + 1):
        for ind in range(len(eigs)):
            if j == ind:
                continue
            eig_gap = abs(eigs[j] - eigs[ind])
            if min_eig_gap is None or eig_gap < min_eig_gap:
                min_eig_gap = eig_gap
    thresh = r * (d*c)**0.5 + 2*c

    print(min_eig_gap, thresh)
    # large eigenvalue gap
    if min_eig_gap > thresh:
        print('anisotropic reached')
        cov = calc_cov_large_gap(d, c, sec_v, sec_beta, eigs, u)
    else:
        cov = calc_cov_small_gap(d, c, sec_v, eigs)
    return cov

# DECISION TREES 

class Node(object):

    def __init__(self, level):
        self.level = level
        self.terminal = False

    def set_split(self, feature, value):
        self.split_condition = (feature, value)

class DecisionTree(object):

    def __init__(self, ordered_features):
        self.ordered_features = ordered_features
        self.root = Node(level=0)
        
    def get_tree_length(self):
         return 2**len(self.ordered_features) - 1

    def calc_entropy(self, y_vals):
        counts = Counter(y_vals)
        total = len(y_vals)
        prs = [k / total for k in counts.values()]
        return entropy(np.array(prs))

    def get_split_measure(self, y_left, y_right):
        total_data = len(y_left + y_right)
        weight_left = len(y_left) / total_data
        weight_right = len(y_right) / total_data        
        return weight_left*self.calc_entropy(y_left) + weight_right* self.calc_entropy(y_right)
    
    def get_left_right(self, x, y, feat, value):
        left, right = [], []
        for ind, data in enumerate(x):
            if data[feat] < value:
                left.append(y[ind])
            else:
                right.append(y[ind])
        return left, right

#     def calc_best_split(self, X, feat, y):
#         possible_splits = set([X[i][feat] for i in range(len(X))])
#         sorted_split_vals = sorted(list(possible_splits))
# #         print(possible_splits, feat)
#         curr_split = None
#         curr_split_meas = None
#         if len(possible_splits) == 1:
#             return 0
#         split_len, split_width, reg_weight = self.regularize[0], self.regularize[1], self.regularize[2]

#         for ind, val in enumerate(sorted_split_vals):
#             start_ind = ind - int(split_len/2)
#             if start_ind < 0:
#                 start_ind = 0
#             end_ind = start_ind + split_len
#             if end_ind > len(sorted_split_vals):
#                 end_ind = len(sorted_split_vals)
#             if end_ind - split_len >= 0:
#                 start_ind = end_ind - split_len
#             split_range = [sorted_split_vals[i] for i in range(start_ind, end_ind)]
#             avg_split_meas = 0.
#             assert val in split_range
#             for split in split_range:
#                 if split == val:
#                     continue
#                 left, right = self.get_left_right(X, y, feat, split)
#                 split_measure = self.get_split_measure(left, right)
#                 avg_split_meas += split_measure
#             if len(split_range) > 1:
#                 total_len = len(split_range) - 1
#             else:
#                 total_len = len(split_range)
#                 assert reg_weight == 0.
           
#             act_left, act_right = self.get_left_right(X, y, feat, val)
#             act_split_meas = self.get_split_measure(act_left, act_right)

#             tot_split_meas = (1-reg_weight)*act_split_meas + reg_weight * avg_split_meas
#             if curr_split_meas is None or tot_split_meas <= curr_split_meas:
                
#                 if tot_split_meas == curr_split_meas:
#                     if val > curr_split:
#                         continue
#                 curr_split_meas = tot_split_meas
#                 curr_split = val
#         return curr_split

#     def calc_best_split_linear(self, X, feat, y):
#         possible_splits = set([X[i][feat] for i in range(len(X))])
#         sorted_split_vals = sorted(list(possible_splits))
#         curr_split = None
#         curr_split_meas = None
#         if len(possible_splits) <= 1:
#             return 0
#         split_len, split_width, reg_weight = self.regularize[0], self.regularize[1], self.regularize[2]
#         for ind, val in enumerate(sorted_split_vals):
#             split_range = np.linspace(val-split_width, val + split_width, split_len)
#             val_ind = int(split_len / 2)
#             assert abs(split_range[val_ind] - val) < 1e-5
#             avg_split_meas = 0.
#             for split in split_range:
#                 if abs(split - val) < 1e-5:
#                     continue
#                 left, right = self.get_left_right(X, y, feat, split)
#                 split_measure = self.get_split_measure(left, right)
#                 avg_split_meas += split_measure
#             if len(split_range) > 1:
#                 total_len = len(split_range) - 1
#             else:
#                 total_len = len(split_range)
#             avg_split_meas /= total_len

#             act_left, act_right = self.get_left_right(X, y, feat, val)
#             act_split_meas = self.get_split_measure(act_left, act_right)

#             tot_split_meas = (1-reg_weight)*act_split_meas + reg_weight*avg_split_meas

#             if curr_split_meas is None or tot_split_meas <= curr_split_meas:
                
#                 if tot_split_meas == curr_split_meas:
#                     if val > curr_split:
#                         continue
#                 curr_split_meas = tot_split_meas
#                 curr_split = val
#         return curr_split

    def calc_best_split_penalize_norm(self, X, feat, y):
        possible_splits = set([X[i][feat] for i in range(len(X))])
        sorted_split_vals = sorted(list(possible_splits))
        curr_split = None
        curr_split_meas = None
        max_precision, reg_param, weight_orig = self.regularize

        if max_precision is not None:
            min_val, max_val = 0, 1
            possible_splits = np.arange(min_val, max_val+max_precision, max_precision)
            possible_splits = [round(x, 8) for x in possible_splits]
        else:
            possible_splits = sorted_split_vals

        entropies = {}
        for ind, val in enumerate(possible_splits):
            left, right = self.get_left_right(X, y, feat, val)
            split_meas = self.get_split_measure(left, right)
            entropies[ind] = split_meas

        for ind, val in enumerate(possible_splits):

            adj_split_val = 0.
            if weight_orig < 1:
                num_adj = 0
                if ind > 0:
                    adj_split_val += entropies[ind-1]
                    num_adj+=1
                if ind < len(possible_splits)-1:
                    adj_split_val += entropies[ind+1]
                    num_adj += 1
                adj_split_val /= num_adj

            weight_adj = 1 - weight_orig
            curr_meas = (weight_orig)*entropies[ind] + weight_adj*adj_split_val
            tot_split_meas = (1-reg_param)*curr_meas + reg_param*val

            if curr_split_meas is None or tot_split_meas <= curr_split_meas:

                if tot_split_meas == curr_split_meas:
                    if val > curr_split:
                        continue
                curr_split_meas = tot_split_meas
                curr_split = val
        return curr_split


    def calc_best_split_precision(self, X, feat, y):
        possible_splits = set([X[i][feat] for i in range(len(X))])
        sorted_split_vals = sorted(list(possible_splits))
        curr_split = None
        curr_split_meas = None

        if len(possible_splits) <= 1:
            return 0

        split_len, split_width, reg_weight, max_precision = self.regularize
        
        if max_precision is not None:
            assert (2*split_width) / split_len % max_precision < 1e-5
            min_val, max_val = 0, 1
            possible_splits = np.arange(min_val, max_val+max_precision, max_precision)
            possible_splits = [round(x, 8) for x in possible_splits]
        else:
            possible_splits = sorted_split_vals
        split_entropies = {}
        for val in possible_splits:
            left, right = self.get_left_right(X, y, feat, val)
            split_meas = self.get_split_measure(left, right)
            split_entropies[val] = split_meas

        for ind, val in enumerate(possible_splits):
            split_range = np.linspace(val-split_width, val + split_width, split_len)
            split_range = [round(x, 8) for x in split_range]
            val_ind = int(split_len / 2)
            assert abs(split_range[val_ind] - val) < 1e-5
            
            avg_split_meas = 0.
            split_range_len = len(split_range)
            for split in split_range:
                if abs(split - val) < 1e-5:
                    split_range_len -= 1
                    continue
                if split < 0 or split > 1:
                    split_range_len -= 1
                    continue           
                if split in split_entropies:
                    avg_split_meas += split_entropies[split]
                else:
                    print('reached')
                    print(split, split_entropies.keys())
                    assert(False)
                    left, right = self.get_left_right(X, y, feat, split)
                    split_measure = self.get_split_measure(left, right)
                    split_entropies[split] = split_measure
                    avg_split_meas += split_measure
            if split_range_len > 0:
                avg_split_meas /= split_range_len
            else:
                avg_split_meas = 10000 # some default large value
                assert reg_weight == 0
            
            act_split_meas = split_entropies[val]
            
            tot_split_meas = (1-reg_weight)*act_split_meas + reg_weight*avg_split_meas + 0.1*val

            if curr_split_meas is None or tot_split_meas <= curr_split_meas:

                if tot_split_meas == curr_split_meas:
                    if val > curr_split:
                        continue
                curr_split_meas = tot_split_meas
                curr_split = val
        return curr_split

    def create_split(self, X, feat, y):
        if len(X) < 1:
            return [], [], [], [], 0 # default split
        value = self.calc_best_split_penalize_norm(X, feat, y) # default to best split
        left_data, right_data = [], []
        left_y, right_y = [], []
        for ind, data_point in enumerate(X):
            if data_point[feat] < value:
                left_data.append(data_point)
                left_y.append(y[ind])
            else:
                right_data.append(data_point)
                right_y.append(y[ind])
        return left_data, right_data, left_y, right_y, value


    def create_tree(self, X, y, regularize=False, curr_node=None):
        self.regularize=regularize
        if curr_node is None:
            curr_node = self.root
        curr_level = curr_node.level
        if curr_level == len(self.ordered_features):
            curr_node.terminal = True
            if len(X) == 0:
                curr_node.classification_value = 0 # default
                curr_node.count = len(y)
            else:
                curr_node.classification_value = max(set(y), key = y.count)
                curr_node.count = len(y)
            return

        feature = self.ordered_features[curr_level]
        left_data, right_data, left_y, right_y, value = self.create_split(X, feature, y)
        curr_node.set_split(feature, value)
        curr_node.left = Node(level = curr_level + 1)
        curr_node.right = Node(level = curr_level + 1)
        self.create_tree(left_data, left_y, regularize=regularize, curr_node=curr_node.left)
        self.create_tree(right_data, right_y, regularize = regularize, curr_node=curr_node.right)
        
    def classify(self, x):
        curr_node = self.root
        while not curr_node.terminal:
            feature, value = curr_node.split_condition
            if x[feature] < value:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right
        return curr_node.classification_value

    def calculate_accuracy(self, X, y):
        correct = 0
        for ind, x_val in enumerate(X):
            pred_y = self.classify(x_val)
            if pred_y == y[ind]:
                correct += 1
        return correct / len(X)

    def ordered_traversal(self, node=None, print_tree=False):
        all_vals, left_vals, right_vals = [], [], []
        value = []
        if node is None:
            node = self.root            
        if hasattr(node, 'split_condition'):
            assert hasattr(node, 'left')
            assert hasattr(node, 'right')
            left_vals = self.ordered_traversal(node.left, print_tree=print_tree)
            feature, value = node.split_condition
            value = [value]
            if print_tree:
                print(f'node at level {node.level} splits on {feature} at {value}')
            right_vals = self.ordered_traversal(node.right, print_tree=print_tree)
        else:
            assert node.terminal == True
            value = [node.classification_value]
            if print_tree:
                print(f'classified as {value[0]} with {node.count} elements')
        all_vals.extend(left_vals)
        all_vals.extend(value)
        all_vals.extend(right_vals)
        if node == self.root:
            assert len(all_vals) == 2**(len(self.ordered_features)+1) - 1
            if print_tree:
                print('--------')
        return all_vals

    def add_noise(self, noise, node=None):
        all_nodes, left_nodes, right_nodes = [], [], []
        value = []
        if node is None:
            node = self.root
        if hasattr(node, 'left'):
            left_nodes = self.add_noise(noise, node.left)
        if hasattr(node, 'split_condition'):
            value = [node]
        if hasattr(node, 'right'):
            right_nodes = self.add_noise(noise, node.right)
        all_nodes.extend(left_nodes)
        all_nodes.extend(value)
        all_nodes.extend(right_nodes)

        if node == self.root:
            assert len(all_nodes) == 2**len(self.ordered_features) - 1
            for ind in range(len(all_nodes)):
                orig_feat, orig_value = all_nodes[ind].split_condition
                all_nodes[ind].split_condition = (orig_feat, orig_value + np.random.normal(0, scale=noise))
        return all_nodes

    def add_noise_aniso(self, noise, node=None):
        all_nodes, left_nodes, right_nodes = [], [], []
        value = []
        if node is None:
            node = self.root
        if hasattr(node, 'left'):
            left_nodes = self.add_noise_aniso(noise, node.left)
        if hasattr(node, 'split_condition'):
            value = [node]
        if hasattr(node, 'right'):
            right_nodes = self.add_noise_aniso(noise, node.right)
        all_nodes.extend(left_nodes)
        all_nodes.extend(value)
        all_nodes.extend(right_nodes)

        if node == self.root:
            assert len(all_nodes) == 2**len(self.ordered_features) - 1
            for ind in range(len(all_nodes)):
                orig_feat, orig_value = all_nodes[ind].split_condition
                all_nodes[ind].split_condition = (orig_feat, orig_value + np.random.normal(0, scale=noise[ind]))
        return all_nodes
    

class Forest(object):
    
    def __init__(self, trees, train_x, train_y):
        self.trees = trees
        self.weights = self.weight_trees(train_x, train_y)
        
    def classify(self, x):
        votes = {}
        for ind, tree in enumerate(self.trees):
            curr_vote = tree.classify(x)
            if curr_vote not in votes:
                votes[curr_vote] = 0

            votes[curr_vote] += self.weights[ind]
        return max(votes, key = votes.get)
    
    def calculate_accuracy(self, X, y):
        correct = 0
        for ind, x_val in enumerate(X):
            pred_y = self.classify(x_val)
            if pred_y == y[ind]:
                correct += 1
        return correct / len(X)
    
    def weight_trees(self, X, y):
        return [round(1/(len(self.trees)), 2) for _ in range(len(self.trees))]

    def add_noise(self, noise):
        for i in range(len(self.trees)):
            self.trees[i].add_noise(noise)
        return

    def add_noise_aniso(self, noise):
        for i in range(len(self.trees)):
            self.trees[i].add_noise_aniso(noise)
        return


class BoostedTrees(object):

    def __init__(self, num_trees, train_x, train_y, ordered_feats, regularize):
        assert len(ordered_feats) == num_trees
        trees = []
        current_train_y = copy.deepcopy(train_y)
        self.regularize=regularize
        for i in range(len(ordered_feats)):
            dt = DecisionTree(ordered_feats[i])
            dt.create_tree(train_x, current_train_y, regularize=regularize)
            trees.append(dt)
            current_train_y = self.fit_residuals(trees, train_x, train_y)
        self.trees = trees
        
    
    def fit_residuals(self, trees, train_x, train_y):
        residual_y = []
        for ind, x in enumerate(train_x):
            est_y = 0.
            for t in trees:
                est_y += t.classify(x)
            residual_y.append(train_y[ind] - est_y)
        return residual_y
        
    def classify(self, x):
        est_val = 0.
        for ind, tree in enumerate(self.trees):
            curr_vote = tree.classify(x)
            est_val += curr_vote

        return round(est_val)
    
    def calculate_accuracy(self, X, y):
        correct = 0
        for ind, x_val in enumerate(X):
            pred_y = self.classify(x_val)
            if pred_y == y[ind]:
                correct += 1
        return correct / len(X)
    
    
def fit_decision_tree(X_train, y_train):
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    feat_imps = clf.feature_importances_
    ordered_feats = [k[0] for k in sorted(
        [(ind, feat_imps[ind]) for ind in range(len(feat_imps))], key = lambda x: x[1], reverse=True)]
    dt = DecisionTree(ordered_feats)
    dt.create_tree(X_train, y_train)
    return dt, dt.ordered_traversal()

def get_ordered_feats(num_feats, num_trees, depth, seed):
    rng = np.random.default_rng(seed=seed)
    feats_list = []
    ordered_feats = list(range(num_feats))
    assert num_trees >= 1
    feats_list = [ordered_feats for i in range(num_trees)]
    
    feats_list = rng.permuted(feats_list, axis=1)
    feats_list = feats_list[:, :depth]
    return feats_list

def fit_forest(train_x, train_y, num_trees, depth, seed, regularize):
    tree_feats = get_ordered_feats(len(train_x[0]), num_trees, depth, seed)
    trees = []
    for i in range(len(tree_feats)):
        dt = DecisionTree(tree_feats[i])
        dt.create_tree(train_x, train_y, regularize=regularize)
        trees.append(dt)
    forest = Forest(trees, train_x, train_y)
    
    traversal = []
    for ind, tree in enumerate(forest.trees):
        traversal.append(forest.weights[ind])
        traversal.extend(tree.ordered_traversal())
    return forest, traversal


def fit_gbdt(train_x, train_y, num_trees, depth, seed, regularize):
    tree_feats = get_ordered_feats(len(train_x[0]), num_trees, depth, seed)
    gbdt = BoostedTrees(num_trees, train_x, train_y, tree_feats, regularize=regularize)
    
    traversal = []
    for ind, tree in enumerate(gbdt.trees):
        traversal.extend(tree.ordered_traversal())
    return gbdt, traversal

# K MEANS 

def infer_cluster_labels(model, actual_labels):
    inferred_labels = {}
    

    for i in range(model.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(model.labels_ == i)
        

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])
        
        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0], minlength=model.n_clusters)
        else:
            counts = np.bincount(np.squeeze(labels), minlength=model.n_clusters)
        sorted_counts = np.argsort(counts)[::-1]

        ind = 0
        while sorted_counts[ind] in inferred_labels:
            ind += 1
        inferred_labels[sorted_counts[ind]] = i

    return inferred_labels

def run_kmeans(train_x, train_y, num_clusters, seed):
    rand_state = np.random.RandomState(seed)
    model = KMeans(n_clusters=num_clusters, random_state=rand_state, n_init="auto").fit(train_x)
    
    centers = model.cluster_centers_
    assert len(centers) == num_clusters
    mapping = infer_cluster_labels(model, train_y)
    
    ordered_centers = []
    new_centers = []
    
    for ind in range(len(centers)):
        index = mapping[ind]
        curr_center = centers[index]
        new_centers.append(curr_center)
        for k in curr_center:
            ordered_centers.append(k)
    model.cluster_centers_ = np.array(new_centers)
    return model, ordered_centers

# SVM

def run_svm(train_x, train_y, num_classes, seed, regularize):
    rand_state = np.random.RandomState(seed)
    model = svm.LinearSVC(dual=False, random_state=rand_state, C=regularize)
    model.fit(train_x, train_y)
    int_shape = (model.intercept_.shape[0], 1)
    int_shaped = np.reshape(model.intercept_, int_shape)
    num_features = model.coef_.shape[1]  
    
    if num_classes == 2: # binary case only has one hyperplane
        tot = np.array(model.coef_[0])
        intercept = model.intercept_[0]
    
    else:
    
        tot = np.zeros((num_classes, num_features))
        intercept = np.zeros((num_classes, 1))
        for ind, c in enumerate(model.classes_):
            tot[c] = model.coef_[ind]
            intercept[c] = model.intercept_[ind]
    tot = np.hstack((tot, intercept))
    tot = tot.flatten()
    return model, tot

# PCA

def gen_pca_data(x_data, end_dim = 10):
    init_dim = len(x_data[0])
    assert end_dim % init_dim == 0
    mult_factor = int(end_dim / init_dim)
    expanded_pts = []
    for pt in x_data:
        expanded = []
        for ind in range(mult_factor):
            if ind == 0:
                noisy_pt = pt
            else:
                noisy_pt = pt + np.random.normal(0, 1e-2, init_dim)
            expanded.extend(noisy_pt)
        expanded_pts.append(expanded)
    return np.array(expanded_pts)

def run_pca(train_x, train_y, num_dims):
    model = PCA(n_components=num_dims)
    model.fit(train_x)
    return model, model.components_

# GENERATE DATA (SYNTHETIC)

def gen_synthetic(num_train=10000, num_test=3000, normalize=False):
    num_features = 2
    cluster_std = 0.55
    num_test_samples = num_test
    centers = np.array([[1, -1], [1, 1], [-1, 1], [-1, -1]])
    syn_num_classes = len(centers)


    syn_train_samples = num_train
    syn_num_classes = len(centers)
    syn_train_x, syn_train_y, centers = make_blobs(n_samples=syn_train_samples,
                                           centers=centers, n_features=num_features, random_state=0,
                                           cluster_std=cluster_std, return_centers=True)

    syn_test_x, syn_test_y = make_blobs(n_samples=num_test_samples, centers=centers, n_features=num_features,
                                random_state=0, cluster_std=cluster_std)
    min_max_scaler = preprocessing.MinMaxScaler()
    syn_train_x = min_max_scaler.fit_transform(syn_train_x)
    syn_test_x = min_max_scaler.transform(syn_test_x)
    return syn_train_x, syn_train_y, syn_test_x, syn_test_y, syn_num_classes, syn_train_samples

def gen_syn_high_dim(dim, num_train=10000, num_test=3000, normalize=False):
    cluster_std = 0.05
    num_test_samples = num_test
    items = [-1, 1]

    centers = []
    for item in product(items, repeat=dim):
        centers.append(list(item))
    centers = np.array(centers)
    syn_num_classes = len(centers)


    syn_train_samples = num_train
    syn_num_classes = len(centers)
    syn_train_x, syn_train_y, centers = make_blobs(n_samples=syn_train_samples,
                                           centers=centers, n_features=dim, random_state=0,
                                           cluster_std=cluster_std, return_centers=True)

    syn_test_x, syn_test_y = make_blobs(n_samples=num_test_samples, centers=centers, n_features=dim,
                                random_state=0, cluster_std=cluster_std)
    min_max_scaler = preprocessing.MinMaxScaler()
    syn_train_x = min_max_scaler.fit_transform(syn_train_x)
    syn_test_x = min_max_scaler.transform(syn_test_x)
    return syn_train_x, syn_train_y, syn_test_x, syn_test_y, syn_num_classes, syn_train_samples

def gen_iris(normalize=False):
    iris = fetch_ucirepo(id=53)
    # data (as pandas dataframes) 
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled = min_max_scaler.fit_transform(iris.data.features)
        X = pd.DataFrame(scaled)
    else:
        X = iris.data.features
    y = iris.data.targets
    all_x = X.to_numpy()
    y_vals = {'Iris-setosa': 0 , 'Iris-versicolor': 1, 'Iris-virginica': 2}
    target_dict = y.to_dict('index')

    y_vec = []
    for ind in range(150):
        y_vec.append(y_vals[target_dict[ind]['class']])

    y_vec = np.array(y_vec)

    all_x, y_vec = shuffle(all_x, y_vec, random_state=0)

    iris_train_x = all_x[:100]
    iris_test_x = all_x[101:]

    iris_train_y = np.array(y_vec[:100])
    iris_test_y = y_vec[101:]
    iris_num_classes = 3

    iris_train_samples = 100
    return iris_train_x, iris_train_y, iris_test_x, iris_test_y, iris_num_classes, iris_train_samples

def gen_obesity(normalize=False):

    df = pd.read_csv('obesity.csv')

    df['Gender'].replace(['Female', 'Male'],
                            [0, 1], inplace=True)

    df['family_history_with_overweight'].replace(['yes', 'no'],
                            [0, 1], inplace=True)

    df['FAVC'].replace(['yes', 'no'],
                            [0, 1], inplace=True)

    df['CAEC'].replace(['no', 'Sometimes', 'Frequently', 'Always'],
                            [0, 1, 2, 3], inplace=True)

    df['SMOKE'].replace(['yes', 'no'],
                            [0, 1], inplace=True)

    df['SCC'].replace(['yes', 'no'],
                            [0, 1], inplace=True)

    df['CALC'].replace(['no', 'Sometimes', 'Frequently', 'Always'],
                            [0, 1, 2, 3], inplace=True)

    df['MTRANS'].replace(['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'],
                            [0, 1, 2, 3, 4], inplace=True)

    df['NObeyesdad'].replace(['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I',
                              'Obesity_Type_II', 'Obesity_Type_III'],
                            [0, 0, 1, 1, 2, 2, 2], inplace=True)
    cols = df.columns
    if normalize:
        cols = df.columns[df.columns != 'NObeyesdad']
        min_max_scaler = preprocessing.MinMaxScaler()
        df[cols] = min_max_scaler.fit_transform(df[cols])

    mat = df.to_numpy()

    y = mat[:, -1]
    x = mat[:, :-1]

    all_x, y_vec = shuffle(x, y)

    train_length = int(x.shape[0]*0.8)

    obesity_train_x = all_x[:train_length]
    obesity_test_x = all_x[train_length+1:]

    obesity_train_y = np.array([int(k) for k in y_vec[:train_length]])
    obesity_test_y = np.array([int(k) for k in y_vec[train_length+1:]])

    obesity_num_classes = 3

    obesity_train_samples = train_length
    return obesity_train_x, obesity_train_y, obesity_test_x, obesity_test_y, obesity_num_classes, obesity_train_samples

def gen_rice(normalize=False):
    rice = fetch_ucirepo(id=545)
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled = min_max_scaler.fit_transform(rice.data.features)
        X = pd.DataFrame(scaled)
    else:
        X = rice.data.features
    y = rice.data.targets
    all_x = X.to_numpy()
    y_vals = {'Cammeo': 0 , 'Osmancik': 1,}
    target_dict = y.to_dict('index')

    y_vec = []
    max_ind = 3810
    for ind in range(max_ind):
        y_vec.append(y_vals[target_dict[ind]['Class']])

    y_vec = np.array(y_vec)

    all_x, y_vec = shuffle(all_x, y_vec)

    train_len = int(0.7*max_ind)
    train_x = all_x[:train_len]
    test_x = all_x[train_len+1:]

    train_y = np.array(y_vec[:train_len])
    test_y = y_vec[train_len+1:]
    num_classes = 2

    return train_x, train_y, test_x, test_y, num_classes, train_len

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def gen_cifar10(normalize=False):
    fnames = ['code/cifar-10-batches-py/data_batch_{}'.format(i) for i in range(1, 6)]
    train_x = []
    train_y = []
    for f in fnames:
        data = unpickle(f)
        train_x.extend(data[b'data'])
        train_y.extend(data[b'labels'])
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_data = 'cifar-10-batches-py/test_batch'
    test_data = unpickle(f)
    test_x = data[b'data']
    test_y = data[b'labels']
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    num_classes = 10
    train_len = train_x.shape[0]

    return train_x, train_y, test_x, test_y, num_classes, train_len

def gen_spam(normalize=False):
    spambase = fetch_ucirepo(id=94) 
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled = min_max_scaler.fit_transform(spambase.data.features)
        X = pd.DataFrame(scaled)
    else:
        X = spambase.data.features
    y = spambase.data.targets
    all_x = X.to_numpy()
    y_vals = {0: 0 , 1: 1,}
    target_dict = y.to_dict('index')
    y_vec = []
    max_ind = 4601
    for ind in range(max_ind):
        y_vec.append(y_vals[target_dict[ind]['Class']])

    y_vec = np.array(y_vec)

    all_x, y_vec = shuffle(all_x, y_vec)

    train_len = int(0.7*max_ind)
    train_x = all_x[:train_len]
    test_x = all_x[train_len+1:]

    train_y = np.array(y_vec[:train_len])
    test_y = y_vec[train_len+1:]
    num_classes = 2

    return train_x, train_y, test_x, test_y, num_classes, train_len

def gen_bean(normalize=False):
    # fetch dataset 
    dry_bean = fetch_ucirepo(id=602) 

    # data (as pandas dataframes) 
    X = dry_bean.data.features 
    y = dry_bean.data.targets
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled = min_max_scaler.fit_transform(dry_bean.data.features)
        X = pd.DataFrame(scaled)
    else:
        X = dry_bean.data.features
    all_x = X.to_numpy()
    target_dict = y.to_dict('index')
    class_names = {'SIRA': 0, 'HOROZ': 1, 'BOMBAY': 2, 'DERMASON': 3, 'SEKER': 4, 'BARBUNYA': 5, 'CALI': 6}
    y_vec = []
    max_ind = 13611
    for ind in range(max_ind):
        y_vec.append(class_names[target_dict[ind]['Class']])

    y_vec = np.array(y_vec)

    all_x, y_vec = shuffle(all_x, y_vec)

    train_len = int(0.7*max_ind)
    train_x = all_x[:train_len]
    test_x = all_x[train_len+1:]

    train_y = np.array(y_vec[:train_len])
    test_y = y_vec[train_len+1:]
    num_classes = len(class_names)

    return train_x, train_y, test_x, test_y, num_classes, train_len
