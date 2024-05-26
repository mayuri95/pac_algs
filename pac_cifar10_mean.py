from algs_lib import *

def hybrid_noise_auto_ind(train_x, train_y, subsample_rate, num_classes,
    eta, regularize=None, num_trees=None, tree_depth = None, max_mi = 0.5, num_dims = None):
    curr_est = None
    converged = False
    curr_trial = 0

    if num_classes is None:
        num_classes = len(set(train_y))

    assert subsample_rate >= num_classes

    est_y = {}
    prev_ests = None

    s1 = None # only relevant for PCA
    
    max_noises = {}
    
    for ind in range(len(train_x)):
        print(f"ind = {ind}")
        removed_train_x = np.delete(train_x, ind, 0)
        removed_train_y = np.delete(train_y, ind, 0)
        while not converged:
            shuffled_x, shuffled_y = shuffle(removed_train_x, removed_train_y)

            shuffled_x, shuffled_y = get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate)
            
            added_x = copy.deepcopy(shuffled_x)
            added_y = copy.deepcopy(shuffled_y)
            added_x[0] = train_x[ind]
            added_y[0] = train_y[ind]

            output_orig = np.average(shuffled_x, axis=0)
            output_new = np.average(added_x, axis=0)
            output = abs(output_orig - output_new)

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
        sqrt_total_var = sum([fin_var[x]**0.5 for x in fin_var])
        for ind in fin_var:
            noise[ind] = 1./(2*max_mi) * fin_var[ind]**0.5 * sqrt_total_var
        for ind in noise:
            if ind not in max_noises or max_noises[ind] < noise[ind]:
                max_noises[ind] = noise[ind]
    return max_noises

# PAC MEAN
train_x, train_y, test_x, test_y, num_classes, train_len = gen_cifar10(normalize=True)
true_mean = np.average(train_x, axis=0)


norms = [np.linalg.norm(x) for x in train_x]
# print(max(norms))

subsample_rate = int(0.5*train_len)

pac_dists = {}
num_trials = 1000
for mi in mi_range:
    scaled_noise = {k: noise[k] * (0.5 / mi) for k in noise}
    iso_noise = max(scaled_noise.values())
    iso_scaled = {k: iso_noise for k in noise}
    avg_dist_pac = 0
    avg_iso_dist_pac = 0
    subsampled_dist = 0
    clipped_train_x = [clip_to_threshold(train_x[i], clip_budget) for i in range(len(train_x))]
    noise = hybrid_noise_auto_ind(clipped_train_x, train_y, subsample_rate, num_classes, 1e-6)
    for _ in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(clipped_train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
        released_mean = np.average(shuffled_x1, axis=0)
        subsampled_dist += np.linalg.norm(released_mean - true_mean)
        for ind in range(len(released_mean)):
            c = np.random.normal(0, scale=scaled_noise[ind])
            released_mean[ind] += c
        avg_dist_pac += np.linalg.norm(released_mean - true_mean)
    subsampled_dist /= num_trials
    print(f'subsampled_dist = {subsampled_dist}')
    for _ in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(clipped_train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
        released_mean = np.average(shuffled_x1, axis=0)
        for ind in range(len(released_mean)):
            c = np.random.normal(0, scale=iso_scaled[ind])
            released_mean[ind] += c
        avg_iso_dist_pac += np.linalg.norm(released_mean - true_mean)
    
    avg_iso_dist_pac /= num_trials
    avg_dist_pac /= num_trials
    print(avg_dist_pac)
    print('-----')
#     subsampled_dist /= num_trials
    pac_dists[mi] = (subsampled_dist, avg_dist_pac, avg_iso_dist_pac)
print(pac_dists)


with open('hybrid_data/pac_cifar10_mean.pkl', 'wb') as f:
    pickle.dump(pac_dists, f)