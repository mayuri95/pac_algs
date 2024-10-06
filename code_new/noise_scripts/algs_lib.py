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
from imblearn.over_sampling import SMOTE


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

def hybrid_noise_test(train_x, train_y, mechanism, subsample_rate, num_classes,
    eta, regularize=None, rebalance = False, num_trees=None, tree_depth = None, max_mi = 0.5, num_dims = None,
    record_ys = False, fname = None):

    avg_dist = 0.
    curr_est = None
    curr_trial = 0
    num_trials = 1024

    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes

    est_y = {}
    est_x = {}
    prev_ests = None
    # 10*c*v
    seed = 743895091 # randomly generated seed for reproducibility
    s1 = None # only relevant for PCA

    while curr_trial < num_trials:
        shuffled_inds = shuffle(list(range(len(train_x))))
        # shuffled_x, shuffled_y = shuffle(train_x, train_y)
        x, y = train_x[shuffled_inds], train_y[shuffled_inds]
        x1, y1 = x[:subsample_rate], y[:subsample_rate]
        x2, y2 = x[subsample_rate:], y[subsample_rate:]
        inds_1, inds_2 = shuffled_inds[:subsample_rate], shuffled_inds[subsample_rate:]

        for (shuffled_x, shuffled_y, shuffled_inds) in [(x1, y1, inds_1), (x2, y2, inds_2)]:
            est_x[curr_trial] = shuffled_inds

            if mechanism == run_kmeans:
                output = mechanism(shuffled_x, shuffled_y, num_classes, seed, rebalance=rebalance)[1]

            if mechanism == run_svm:
                output = mechanism(shuffled_x, shuffled_y, num_classes, seed, regularize)[1]

            if mechanism.__name__ == 'fit_forest' or mechanism.__name__ == 'fit_gbdt':
                assert num_trees is not None
                assert tree_depth is not None
                output = mechanism(shuffled_x, shuffled_y, num_trees, tree_depth, seed, regularize)[1]


            if mechanism.__name__ == 'run_pca':
                assert num_dims is not None
                output = mechanism(shuffled_x, shuffled_y, num_dims, seed)[1]
                if s1 is None:
                    s1 = output
                    output = output.flatten()
                else:
                    s2 = output
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
                    output = s2

            for ind in range(len(output)):
                if ind not in est_y:
                    est_y[ind] = []
                est_y[ind].append(output[ind])

            curr_trial += 1
    fin_var = {ind: np.var(est_y[ind]) for ind in est_y}

    noise = {}
    sqrt_total_var = sum([fin_var[x]**0.5 for x in fin_var])
    for ind in fin_var:
        noise[ind] = 1./(2*max_mi) * fin_var[ind]**0.5 * sqrt_total_var
    if record_ys:
        assert fname is not None
        with open(fname, 'wb') as f:
            pickle.dump(est_y, f)
    return est_x, noise


def hybrid_noise_mean_ind(train_x, train_y, subsample_rate, num_classes,
    eta, regularize=None, num_trees=None, tree_depth = None, max_mi = 0.5, num_dims = None):

    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes


    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes

    est_y = {}
    est_x = {}
    prev_ests = None
    seed = 743895091 # randomly generated seed for reproducibility
    num_trials = 1024
    curr_trial = 0
    max_noises = {}

    while curr_trial < num_trials:
        shuffled_inds = shuffle(list(range(len(train_x))))
        # shuffled_x, shuffled_y = shuffle(train_x, train_y)
        x, y = train_x[shuffled_inds], train_y[shuffled_inds]
        x1, y1 = x[:subsample_rate], y[:subsample_rate]
        x2, y2 = x[subsample_rate:], y[subsample_rate:]
        inds_1, inds_2 = shuffled_inds[:subsample_rate], shuffled_inds[subsample_rate:]
        est_x[curr_trial] = inds_1
        est_x[curr_trial + 1] = inds_2
        curr_trial += 2

    
    
    for ind in range(len(train_x)):
        print(f'ind is {ind}')

        xs_in = [k for k in est_x if ind in est_x[k]]
        xs_out = [k for k in est_x if ind not in est_x[k]]

        for set_ind in xs_in:
            s_x = set(est_x[set_ind])
            output_orig = np.average(train_x[est_x[set_ind]], axis=0)

            best_match, best_dist = None, None
            for k in xs_out:
                indices = est_x[k]
                output_new = np.average(train_x[est_x[k]], axis=0)
                dist = np.linalg.norm(output_orig - output_new)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_match = k

            output_orig = np.average(train_x[est_x[set_ind]], axis=0)
            output_new = np.average(train_x[est_x[best_match]], axis=0)
            output = (output_orig - output_new)**2

            for ind in range(len(output)):
                if ind not in est_y:
                    est_y[ind] = []
                est_y[ind].append(output[ind])

        fin_var = {ind: np.average(est_y[ind]) for ind in est_y}

        noise = {}
        sqrt_total_var = sum([fin_var[x]**0.5 for x in fin_var])
        for ind in fin_var:
            noise[ind] = 1./(2*max_mi) * fin_var[ind]**0.5 * sqrt_total_var
        for ind in noise:
            if ind not in max_noises or max_noises[ind] < noise[ind]:
                max_noises[ind] = noise[ind]
    return est_x, max_noises

def hybrid_noise_mean(train_x, train_y, subsample_rate, num_classes,
    eta, regularize=None, num_trees=None, tree_depth = None, max_mi = 0.5, num_dims = None):


    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes

    est_y = {}
    est_x = {}
    prev_ests = None
    seed = 743895091 # randomly generated seed for reproducibility
    num_trials = 1024
    curr_trial = 0

    while curr_trial < num_trials:
        shuffled_inds = shuffle(list(range(len(train_x))))
        # shuffled_x, shuffled_y = shuffle(train_x, train_y)
        x, y = train_x[shuffled_inds], train_y[shuffled_inds]
        x1, y1 = x[:subsample_rate], y[:subsample_rate]
        x2, y2 = x[subsample_rate:], y[subsample_rate:]
        inds_1, inds_2 = shuffled_inds[:subsample_rate], shuffled_inds[subsample_rate:]

        for (shuffled_x, shuffled_y, shuffled_inds) in [(x1, y1, inds_1), (x2, y2, inds_2)]:
            est_x[curr_trial] = shuffled_inds
            output = np.average(shuffled_x, axis=0)

            for ind in range(len(output)):
                if ind not in est_y:
                    est_y[ind] = []
                est_y[ind].append(output[ind])

            if curr_trial % 10 == 0:
                if curr_trial % 100 == 0:
                    print(f'curr trial is {curr_trial}')
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
    sqrt_total_var = sum([fin_var[x]**0.5 for x in fin_var])
    for ind in fin_var:
        noise[ind] = 1./(2*max_mi) * fin_var[ind]**0.5 * sqrt_total_var
    return est_x, noise


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
    
    def get_left_right(self, x, y, feat, value, include_x = False):
        left, right = [], []
        xleft, xright = [], []
        for ind, data in enumerate(x):
            if data[feat] < value:
                xleft.append(data)
                left.append(y[ind])
            else:
                xright.append(data)
                right.append(y[ind])
        if not include_x:
            return left, right
        return xleft, left, xright, right

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
            if weight_orig < 1 and len(entropies) > 1:
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
        
    def remake_classes(self, X, y, node=None):
        if node is None:
            node = self.root
        if hasattr(node, 'split_condition'):
            assert hasattr(node, 'left')
            assert hasattr(node, 'right')
            xleft, yleft, xright, yright = self.get_left_right(X, y, node.split_condition[0], node.split_condition[1], include_x=True)
            self.remake_classes(xleft, yleft, node=node.left)
            self.remake_classes(xright, yright, node=node.right)
        else:
            assert node.terminal == True
            if len(y) == 0:
                node.classification_value = 0 # default
                node.count = len(y)
            else:
                node.classification_value = max(set(y), key = y.count)
                node.count = len(y)

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

    def ordered_traversal(self, node=None, print_tree=False, include_classes=False):
        all_vals, left_vals, right_vals = [], [], []
        value = []
        if node is None:
            node = self.root            
        if hasattr(node, 'split_condition'):
            assert hasattr(node, 'left')
            assert hasattr(node, 'right')
            left_vals = self.ordered_traversal(node.left, print_tree=print_tree, include_classes=include_classes)
            feature, value = node.split_condition
            value = [value]
            if print_tree:
                assert include_classes==True
                print(f'node at level {node.level} splits on {feature} at {value}')
            right_vals = self.ordered_traversal(node.right, print_tree=print_tree, include_classes=include_classes)
        else:
            assert node.terminal == True
            if include_classes:
                value = [node.classification_value]
            else:
                value = []
            if print_tree:
                assert include_classes == True
                print(f'classified as {value[0]} with {node.count} elements')
        all_vals.extend(left_vals)
        all_vals.extend(value)
        all_vals.extend(right_vals)
        if node == self.root:
            if include_classes:
                assert len(all_vals) == 2**(len(self.ordered_features)+1) - 1
            else:
                assert len(all_vals) == 2**(len(self.ordered_features)) - 1
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

    def add_noise_aniso(self, noise, train_x, train_y, node=None):
        all_nodes, left_nodes, right_nodes = [], [], []
        value = []
        if node is None:
            node = self.root
        if hasattr(node, 'left'):
            left_nodes = self.add_noise_aniso(noise, train_x, train_y, node=node.left)
        if hasattr(node, 'split_condition'):
            value = [node]
        if hasattr(node, 'right'):
            right_nodes = self.add_noise_aniso(noise, train_x, train_y, node=node.right)
        all_nodes.extend(left_nodes)
        all_nodes.extend(value)
        all_nodes.extend(right_nodes)

        if node == self.root:
            assert len(all_nodes) == 2**len(self.ordered_features) - 1
            for ind in range(len(all_nodes)):
                orig_feat, orig_value = all_nodes[ind].split_condition
                all_nodes[ind].split_condition = (orig_feat, orig_value + np.random.normal(0, scale=noise[ind]))
            #self.remake_classes(train_x, train_y)
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

    def add_noise_aniso(self, noise, x, y):
        for i in range(len(self.trees)):
            self.trees[i].add_noise_aniso(noise, x, y)
        return

def get_ordered_feats(num_feats, num_trees, depth, seed):
    rng = np.random.default_rng(seed=seed)
    feats_list = []
    ordered_feats = list(range(num_feats))
    assert num_trees >= 1
    feats_list = [ordered_feats for i in range(num_trees)]
    
    feats_list = rng.permuted(feats_list, axis=1)
    feats_list = feats_list[:, :depth]
    return feats_list
    
def fit_decision_tree(X_train, y_train):
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    feat_imps = clf.feature_importances_
    ordered_feats = [k[0] for k in sorted(
        [(ind, feat_imps[ind]) for ind in range(len(feat_imps))], key = lambda x: x[1], reverse=True)]
    dt = DecisionTree(ordered_feats)
    dt.create_tree(X_train, y_train)
    return dt, dt.ordered_traversal()

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

def run_kmeans(train_x, train_y, num_clusters, seed, weights = None, rebalance = False):
    rand_state = np.random.RandomState(seed)
    if rebalance:
        sm = SMOTE(random_state=seed)
        train_x, train_y = sm.fit_resample(train_x, train_y)
        model = KMeans(n_clusters=num_clusters, random_state=rand_state,
                   max_iter = 1000, init='k-means++', n_init=10).fit(train_x)
    else:
        model = KMeans(n_clusters=num_clusters, random_state=rand_state, n_init=10).fit(train_x)
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

def run_pca(train_x, train_y, num_dims, seed):
    rand_state = np.random.RandomState(seed)
    model = PCA(n_components=num_dims)
    model.fit(train_x)
    return model, model.components_

# GENERATE DATA (SYNTHETIC)

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

def gen_rice(normalize=False, use_file = False):
    if not use_file:
        rice = fetch_ucirepo(id=545)
        if normalize:
            min_max_scaler = preprocessing.MinMaxScaler()
            scaled = min_max_scaler.fit_transform(rice.data.features)
            X = pd.DataFrame(scaled)
        else:
            X = rice.data.features
        y = rice.data.targets
    else:
        X = pd.read_csv('rice_feats.csv')
        y = pd.read_csv('rice_targets.csv')
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled = min_max_scaler.fit_transform(X)
        X = pd.DataFrame(scaled)
    else:
        X = rice.data.features
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
    fnames = ['cifar-10-batches-py/data_batch_{}'.format(i) for i in range(1, 6)]
    all_x = []
    all_y = []
    for f in fnames:
        data = unpickle(f)
        all_x.extend(data[b'data'])
        all_y.extend(data[b'labels'])
    test_data = 'cifar-10-batches-py/test_batch'
    test_data = unpickle(f)
    all_x.extend(data[b'data'])
    all_y.extend(data[b'labels'])
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        scaled = min_max_scaler.fit_transform(all_x)
        all_x = np.array(pd.DataFrame(scaled))
#     print(all_x.shape)
    train_x = np.array(all_x[:50000, :])
    test_x = np.array(all_x[50000:, :])
#     print(train_x.shape)
#     print(test_x.shape)
    
    train_y = np.array(all_y[:50000])
    test_y = np.array(all_y[50000:])

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

def clip_to_threshold(vec, c):
    curr_norm = np.linalg.norm(vec)
    if curr_norm <= c:
        clip_ratio = 1.0
    clip_ratio = c / curr_norm
    return np.array([vec[i]*clip_ratio for i in range(len(vec))])

def add_noise(scale):
    return np.random.laplace(0, scale)
# global sensitivity is C/n i think?
# so scale should be (C/n) / \epsilon per elem?

def calc_posterior(mi, prior=0.5, prec = 100000):
    test_vals = [x / prec for x in range(1, prec)]
    max_t = None
    for t in test_vals:
        if t*np.log(t/prior)+(1-t)*np.log((1-t)/(1-prior)) <= mi:
            if  max_t is None or t > max_t:
                max_t = t
    return max_t

def dp_epsilon_to_posterior_success(epsilon):
    return 1 - 1./(1+np.exp(epsilon))

def dp_ps_to_epsilon(ps):
    return np.log(ps / (1-ps))
