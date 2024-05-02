from algs_lib import *

def clip_to_threshold(vec, c):
    curr_norm = np.linalg.norm(vec)
    if curr_norm <= c:
        return vec
    clip_ratio = c / curr_norm
    return [vec[i]*clip_ratio for i in range(len(vec))]

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

def hybrid_noise_auto(train_x, train_y, subsample_rate, num_classes,
    eta, regularize=None, num_trees=None, tree_depth = None, max_mi = 0.5, num_dims = None):

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

    s1 = None # only relevant for PCA
    
    while not converged:
        shuffled_x, shuffled_y = shuffle(train_x, train_y)
        
        shuffled_x, shuffled_y = get_samples_safe(shuffled_x, shuffled_y, num_classes, subsample_rate)
        
        output = np.average(shuffled_x, axis=0)

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
    return noise


def gen_cifar10(normalize=True):
    fnames = ['cifar-10-batches-py/data_batch_{}'.format(i) for i in range(1, 6)]
    train_x = []
    train_y = []
    for f in fnames:
        data = unpickle(f)
        train_x.extend(data[b'data'])
        train_y.extend(data[b'labels'])
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    min_max_scaler = preprocessing.MinMaxScaler()
    scaled = min_max_scaler.fit_transform(train_x)
    train_x = np.array(pd.DataFrame(scaled))

    test_data = 'cifar-10-batches-py/test_batch'
    test_data = unpickle(f)
    test_x = data[b'data']
    test_y = data[b'labels']
    test_x = np.array(min_max_scaler.fit_transform(test_x))
    test_y = np.array(test_y)
    num_classes = 10
    train_len = train_x.shape[0]

    return train_x, train_y, test_x, test_y, num_classes, train_len


train_x, train_y, test_x, test_y, num_classes, train_len = gen_cifar10(normalize=True)
true_mean = np.average(train_x, axis=0)

norms = [np.linalg.norm(x) for x in train_x]
print(max(norms))

print("DP!")

# DP MEAN
dp_dists = {}
num_trials = 200
epsilon_vals = [1.6426117097961406, 0.7304317044395013, 0.3563228120191924]

for eps in epsilon_vals:
    avg_dist_dp = {}
    for i in range(1, 55):
        print(eps, i)
        clip_budget = i
        clipped_train_x = [clip_to_threshold(train_x[i], clip_budget) for i in range(len(train_x))]
        released_mean = np.average(clipped_train_x, axis=0)
        clip_dist = np.linalg.norm(released_mean - true_mean)
        dist = 0.
        for _ in range(num_trials):
            released_mean = np.average(clipped_train_x, axis=0)
            for ind in range(len(released_mean)):
                sensitivity = clip_budget / train_len 
                released_mean[ind] += add_noise(sensitivity / eps)
            dist += np.linalg.norm(released_mean - true_mean)
        dist /= num_trials
        avg_dist_dp[i] = (clip_dist, dist)
    dp_key = min(avg_dist_dp.items(), key=lambda x: x[1][1])[0]
    dp_dists[eps] = avg_dist_dp[dp_key]

print(dp_dists)

print("PAC!")
# PAC MEAN
subsample_rate = int(0.5*train_len)

noise = hybrid_noise_auto(train_x, train_y, subsample_rate, num_classes, 1e-6)

pac_dists = {}
num_trials = 1000

for mi in mi_range:
    scaled_noise = {k: noise[k] * (0.5 / mi) for k in noise}
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
    avg_iso_dist_pac /= num_trials
    avg_dist_pac /= num_trials
    subsampled_dist /= num_trials
    pac_dists[mi] = (subsampled_dist, avg_dist_pac, avg_iso_dist_pac)

print(pac_dists)