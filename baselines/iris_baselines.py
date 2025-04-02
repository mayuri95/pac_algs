from algs_lib import *

num_trees = 1
tree_depth = 3

regs = [(None, 0, 1.0), (0.01, 0.2, 0.8)]
C_vals = [1.0, 1e-6]

seed = 743895091
num_trials = 100

baseline_accs = {}


rebalance = [True, False]
# K MEANS
baseline_accs['kmeans'] = {}
for reb in rebalance:
    avg_acc = 0.
    for _ in range(num_trials)
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind='power')
        model, cluster_centers = run_kmeans(train_x, train_y, num_clusters=num_classes, seed=seed, rebalance=reb)
        predictions = model.predict(test_x)
        acc = accuracy_score(test_y, predictions)
        avg_acc += acc
    baseline_accs['kmeans'][reb] = avg_acc / num_trials

# SVM
baseline_accs['svm'] = {}
for C in C_vals:
    if C == 1.0:
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind='robust')
    else:
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind='power_standard')
    model, svm_vec = run_svm(train_x, train_y, num_classes=num_classes, seed=seed,
                             regularize=C)
    acc = model.score(test_x, test_y)
    baseline_accs['svm'][C] = acc

# DT

baseline_accs['dt'] = {}
for reg in regs:
    if reg == (None, 0, 1.0):
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind='power')
    else:
        train_x, train_y, test_x, test_y, num_classes, train_len = gen_iris(normalize=True, norm_kind='robust')
    forest, forest_vec = fit_forest(train_x, train_y, num_trees, tree_depth, regularize=reg, seed=seed)
    acc = forest.calculate_accuracy(test_x, test_y)
    baseline_accs['dt'][reg] = acc


with open('baselines/iris_baselines.pkl', 'wb') as f:
    pickle.dump(baseline_accs, f)
