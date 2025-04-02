from algs_lib import *
from scipy import stats
from sklearn.decomposition import PCA
import sys

fns = [gen_iris, gen_rice, gen_bean]
name = ['iris', 'rice', 'bean']
arg_ind = int(sys.argv[1])

norm = 'normalizer'
mi_range = [0.25, 1/16., 0.015625]
posterior_success_rates = [calc_posterior(mi) for mi in mi_range]
epsilon_vals = [dp_ps_to_epsilon(ps) for ps in posterior_success_rates]

print(f'data is {name[arg_ind]}')
fn = fns[arg_ind]
train_x, train_y, test_x, test_y, num_classes, train_len = fn(normalize=True, norm_kind=norm)

true_mean = np.average(train_x, axis=0)
subsample_rate = int(0.5*train_len)

all_dists = {}

noise = hybrid_noise_mean_ind(train_x, train_y, subsample_rate, num_classes, 1e-6)
pac_dists = {}
num_trials = 1000
for mi in mi_range:
    scaled_noise = {k: np.sqrt(noise[k] * (0.5 / mi)) for k in noise}
    iso_noise = max(scaled_noise.values())
    iso_scaled = {k: iso_noise for k in noise}
    avg_dist_pac = 0
    avg_iso_dist_pac = 0
    subsampled_dist = 0
    for _ in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
        released_mean = np.average(shuffled_x1, axis=0)
        subsampled_dist += np.linalg.norm(released_mean - true_mean)
        for ind in range(len(released_mean)):
            c = np.random.normal(0, scale=scaled_noise[ind])
            released_mean[ind] += c
        avg_dist_pac += np.linalg.norm(released_mean - true_mean)

    for _ in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
        released_mean = np.average(shuffled_x1, axis=0)
        for ind in range(len(released_mean)):
            c = np.random.normal(0, scale=iso_scaled[ind])
            released_mean[ind] += c
        avg_iso_dist_pac += np.linalg.norm(released_mean - true_mean)

    subsampled_dist /= num_trials
    avg_iso_dist_pac /= num_trials
    avg_dist_pac /= num_trials

    pac_dists[mi] = (float(subsampled_dist), float(avg_dist_pac), float(avg_iso_dist_pac))

all_dists['individual'] = pac_dists
print('individual privacy')
print(pac_dists)


subsample_rate = int(0.5*train_len)
noise = hybrid_noise_mean(train_x, train_y, subsample_rate, num_classes, eta=1e-6)
pac_dists = {}
num_trials = 1000
for mi in mi_range:
    scaled_noise = {k: np.sqrt(noise[k] * (0.5 / mi)) for k in noise}
    iso_noise = max(scaled_noise.values())
    iso_scaled = {k: iso_noise for k in noise}
    avg_dist_pac = 0
    avg_iso_dist_pac = 0
    subsampled_dist = 0
    for _ in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
        released_mean = np.average(shuffled_x1, axis=0)
        subsampled_dist += np.linalg.norm(released_mean - true_mean)
        for ind in range(len(released_mean)):
            c = np.random.normal(0, scale=scaled_noise[ind])
            released_mean[ind] += c
        avg_dist_pac += np.linalg.norm(released_mean - true_mean)

    for _ in range(num_trials):
        shuffled_x1, shuffled_y1 = shuffle(train_x, train_y)
        shuffled_x1, shuffled_y1 = get_samples_safe(shuffled_x1, shuffled_y1, num_classes, subsample_rate)
        released_mean = np.average(shuffled_x1, axis=0)
        for ind in range(len(released_mean)):
            c = np.random.normal(0, scale=iso_scaled[ind])
            released_mean[ind] += c
        avg_iso_dist_pac += np.linalg.norm(released_mean - true_mean)
    subsampled_dist /= num_trials
    avg_iso_dist_pac /= num_trials
    avg_dist_pac /= num_trials
    pac_dists[mi] = (float(subsampled_dist), float(avg_dist_pac), float(avg_iso_dist_pac))
print('global pac privacy')
print(pac_dists)
all_dists['global'] = pac_dists

# DP MEAN

dp_dists = {}
num_trials = 1000
for eps in epsilon_vals:
    avg_dist_dp = {}
    for i in range(1, 101):
        clip_budget = i / 100.
        clipped_train_x = [clip_to_threshold(train_x[i], clip_budget) for i in range(len(train_x))]
        released_mean = np.average(clipped_train_x, axis=0)
        clip_dist = np.linalg.norm(released_mean - true_mean)
        dist = 0.
        sensitivity = clip_budget / train_len
        if sensitivity/eps > 10:
            continue
        for _ in range(num_trials):
            released_mean = np.average(clipped_train_x, axis=0)
            for ind in range(len(released_mean)):
                sensitivity = clip_budget / train_len
                released_mean[ind] += add_noise(sensitivity/eps)
            dist += np.linalg.norm(released_mean - true_mean)
        dist /= num_trials
        avg_dist_dp[i] = (float(clip_dist), float(dist))
    dp_key = min(avg_dist_dp.items(), key=lambda x: x[1][1])[0]
    dp_dists[float(eps)] = avg_dist_dp[dp_key]
    print(dp_key)
all_dists['dp'] = dp_dists
print('DP:')
print([(eps, dp_dists[eps][1]) for eps in dp_dists])

with open(f'mean_dists/global_{name[arg_ind]}_mean_dist.pkl', 'wb') as f:
    pickle.dump(all_dists, f)
