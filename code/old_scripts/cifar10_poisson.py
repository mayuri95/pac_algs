from algs_lib import *


# PAC MEAN
train_x, train_y, test_x, test_y, num_classes, train_len = gen_cifar10(normalize=True)
subsample_rate = int(0.5*train_len)
clip = 29
train_x = np.array([clip_to_threshold(train_x[i], clip) for i in range(len(train_x))])

true_mean = np.average(train_x, axis=0)

var = {}
for i in range(len(train_x)):
    var[i] = max(train_x[:, i]) / train_len

noise = {}
sqrt_total_var = sum([var[x]**0.5 for x in var])
for ind in fin_var:
    noise[ind] = 1./(2*max_mi) * var[ind]**0.5 * sqrt_total_var


print(f'noise norm is {sum(noise.values())}')
norms = [np.linalg.norm(x) for x in train_x]
mi_range = [0.25, 1/16., 1/64.]


pac_dists = {}
num_trials = 1000
for mi in mi_range:
    scaled_noise = {k: noise[k] * (0.5 / mi) for k in noise}
    norm_noise = sum([scaled_noise[k]**2 for k in scaled_noise])**0.5
    print(f'norm of noise is {norm_noise}')
    iso_noise = max(scaled_noise.values())
    iso_scaled = {k: iso_noise for k in noise}
    avg_dist_pac = 0
    avg_iso_dist_pac = 0
    subsampled_dist = 0
    for _ in range(num_trials):
        shuffled_x1, shuffled_y1 = [], []
        for train_ind in range(len(train_x)):
            rand_value = np.random.rand()
            if rand_value >= 0.5:
                shuffled_x1.append(train_x[train_ind])
        released_mean = np.average(shuffled_x1, axis=0)
        subsampled_dist += np.linalg.norm(released_mean - true_mean)
        for ind in range(len(released_mean)):
            c = np.random.normal(0, scale=scaled_noise[ind])
            released_mean[ind] += c
        avg_dist_pac += np.linalg.norm(released_mean - true_mean)
    subsampled_dist /= num_trials
    print(f'subsampled_dist = {subsampled_dist}')
    for _ in range(num_trials):
        shuffled_x1, shuffled_y1 = [], []
        for train_ind in range(len(train_x)):
            rand_value = np.random.rand()
            if rand_value >= 0.5:
                shuffled_x1.append(train_x[train_ind])
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

with open('hybrid_data/cifar10_mean_noise.pkl', 'wb') as f:
    pickle.dump(noise, f)

with open('hybrid_data/cifar10_mean_dist.pkl', 'wb') as f:
    pickle.dump(pac_dists, f)

