from algs_lib import *

regs = [(None, 0, 1.0), (0.05, 0.25, 0.6)]
tree_params = [(3,3), (3,5)]
C_vals = [1.0, 0.001]
dims = [1, 8]

seed = 743895091

baseline_accs = {}


rebalance = [True, False]
# K MEANS
baseline_accs['kmeans'] = {}
for reb in rebalance:
    if reb is True:
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(
            normalize=True, norm_kind='quantile_gaussian')
    else:
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True, norm_kind='minmax')
    model, cluster_centers = run_kmeans(train_x, train_y, num_clusters=num_classes, seed=seed, rebalance=reb)
    predictions = model.predict(test_x)
    acc = accuracy_score(test_y, predictions)
    baseline_accs['kmeans'][reb] = acc

# SVM
baseline_accs['svm'] = {}
for C in C_vals:
    if C == 1.0:
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True, norm_kind='robust')
    else:
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True, norm_kind='standard')
    model, svm_vec = run_svm(train_x, train_y, num_classes=num_classes, seed=seed,
                             regularize=C)
    acc = model.score(test_x, test_y)
    baseline_accs['svm'][C] = acc

# DT

baseline_accs['dt'] = {}
for reg_ind, reg in enumerate(regs):
    if reg == (None, 0, 1.0):
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True, norm_kind='minmax')

    else:
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True, norm_kind='standard_no_std')
    num_trees, tree_depth = tree_params[reg_ind]
    forest, forest_vec = fit_forest(train_x, train_y, num_trees, tree_depth, regularize=reg, seed=seed)
    acc = forest.calculate_accuracy(test_x, test_y)
    baseline_accs['dt'][reg] = acc

# PCA

baseline_accs['pca'] = {}
for dim in dims:
    train_x, train_y, test_x, test_y, num_classes, train_len = gen_bean(normalize=True, norm_kind='minmax')
    model, components = run_pca(train_x, train_y, num_dims=dim, seed=seed)
    predictions = model.inverse_transform(model.transform(test_x))
    acc = np.linalg.norm(test_x - predictions)
    acc /= np.linalg.norm(test_x)
    baseline_accs['pca'][dim] = acc

with open('baselines/bean_baselines.pkl', 'wb') as f:
	pickle.dump(baseline_accs, f)
