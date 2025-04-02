from algs_lib import *

dims = [1, 3]

seed = 743895091

baseline_accs = {}
# PCA

baseline_accs['pca'] = {}
for dim in dims:
    train_x, train_y, test_x, test_y, num_classes, train_len = gen_cifar10(normalize=True, norm_kind='minmax')
    model, components = run_pca(train_x, train_y, num_dims=dim, seed=seed)
    predictions = model.inverse_transform(model.transform(test_x))
    acc = np.linalg.norm(test_x - predictions)
    acc /= np.linalg.norm(test_x)
    baseline_accs['pca'][dim] = acc

with open('baselines/cifar10_baselines.pkl', 'wb') as f:
	pickle.dump(baseline_accs, f)
