import os
import itertools as it
import time
import csv
import autograd.numpy as np
import autograd.numpy.random as npr

npr.seed(0)

from autograd.scipy.misc import logsumexp
from scipy.optimize import linear_sum_assignment
from autograd.util import flatten
from autograd.optimizers import adam, sgd
from autograd import grad

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')
from birkhoff.qap import solve_qap
from birkhoff.primitives import gaussian_entropy, logistic
from birkhoff.utils import cached
from copy import deepcopy
import seaborn as sns

sns.set_context("talk")
sns.set_style("white")

color_names = ["red",
               "windows blue",
               "amber",
               "faded green",
               "dusty purple",
               "orange",
               "clay",
               "pink",
               "greyish",
               "light cyan",
               "steel blue",
               "pastel purple",
               "mint",
               "salmon"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("paper")

### Set the inputs/outputs
CELEGANS_NETWORK = os.path.join("data", "C-elegans-frontal.txt")
CELEGANS_METADATA = os.path.join("data", "C-elegans-frontal-meta.csv")
RESULTS_DIR = os.path.join("results", "celegans_real_network", "2017_05_18")


### Synthetic data
# def load_celegans_network():
#     # Load the network
#     # rows = pre; cols = post
#     A = np.zeros((131,131), dtype=bool)
#     with open(CELEGANS_NETWORK, "rb") as f:
#         for line in f:
#             if line.startswith(b'#'):
#                 continue
#             else:
#                 pres, posts = line.split(b' ')
#                 A[int(pres), int(posts)] = 1
#                 # print("A: {} -> {}".format(int(pres), int(posts)))
#     assert np.all(A.sum() == 764)
#
#     # Load the positions
#     import pandas as pd
#     meta = pd.read_csv(CELEGANS_METADATA)
#     node_id = np.array(meta.node_id)
#     assert np.all(np.diff(node_id) == 1), "Sanity check failed..."
#     names = np.array(meta.name)
#     posx = np.array(meta.posx)
#     posy = np.array(meta.posy)
#
#     # Sort by posx
#     perm = np.argsort(posx)
#     A = A[np.ix_(perm, perm)]
#     names = names[perm]
#     posx = posx[perm]
#     posy = posy[perm]
#
#     return A, names, posx, posy


def load_celegans_network(props = np.ones((3, 4))):
    """" This function loads a connectome with a subsample of the entire connectome. The sub-sample
        is given by props. props[i,j] = proportion of neurons of category (i,j) to include
        category i = body position (Head = 0, Middle =1, Tail =2)
        category j = neuron type (Sensory = 0, Motor = 1, Interneuron =2, Poly-type =3)
        Besides names and positions of neurons, it outputs an array of adjacency matrix, for each type of
        connectivity (Synapse, electric junction and NMJ (?))"""


    NeuronTypeCSV = csv.reader(open('../data/NeuronType.csv', 'r'), delimiter=',', skipinitialspace=True)
    neuron_info_all = [[] for index in range(4)]
    relevant_indexes = [0, 1, 2, 14]
    # load relevant information (names, numerica position, anatomical position and type)
    for row in NeuronTypeCSV:
        for j0, j in enumerate(relevant_indexes):
            neuron_info_all[j0].append(row[j].strip(' \t\n\r'))

    names_with_zeros = deepcopy(neuron_info_all[0])
    # erase extra zeros in name
    for j in range(279):
        indZero = neuron_info_all[0][j].find('0')
        if (indZero >= 0 and indZero < len(neuron_info_all[0][j]) - 1):
            neuron_info_all[0][j] = neuron_info_all[0][j].replace('0', '')

    names = deepcopy(neuron_info_all[0])
    xpos = np.array(neuron_info_all[1])

    location = neuron_info_all[2]

    issensory = np.zeros(279)
    ismotor = np.zeros(279)
    isinterneuron = np.zeros(279)

    NeuronTypeISM = csv.reader(open('../data/NeuronTypeISM.csv', 'r'), delimiter=',', skipinitialspace=True)

    for row in NeuronTypeISM:
        try:
            index = names.index(row[0])
            words = row[2].lower()
            if ('sensory' in words):
                issensory[index] = 1
            if ('motor' in words):
                ismotor[index] = 1
            if ('interneuron' in words):
                isinterneuron[index] = 1
        except:
            pass
    NeuronRemainingTypesISM = csv.reader(open('../data/NeuronRemainingTypesISM.csv', 'r'), delimiter=',',
                                         skipinitialspace=True)
    for row in NeuronRemainingTypesISM:
        try:
            index = neuron_info_all[0].index(row[0])
            words = row[1].lower()
            if ('sensory' in words):
                issensory[index] = 1
            if ('motor' in words):
                ismotor[index] = 1
            if ('interneuron' in words):
                isinterneuron[index] = 1
        except:
            pass

    ConnectomeCSV = csv.reader(open('../data/NeuronConnect.csv', 'r'), delimiter=',', skipinitialspace=True)
    As_weighted = np.zeros((3, 279, 279))

    for row in ConnectomeCSV:
        try:
            index1 = names_with_zeros.index(row[0])
            index2 = names_with_zeros.index(row[1])
            if ('S' in row[2] or 'R' in row[2] or 'Sp' in row[2] or 'Rp' in row[2]):
                As_weighted[0, index1, index2] = As_weighted[0, index1, index2] + float(row[3])
            if ('EJ' in row[2]):
                As_weighted[1, index1, index2] = As_weighted[1, index1, index2] + float(row[3])
            if ('NMJ' in row[2]):
                As_weighted[2, index1, index2] = As_weighted[2, index1, index2] + float(row[3])
        except:
            pass
    As = (As_weighted > 0).astype(int)

    ind_type = [[] for _ in range(4)]

    # 0=sensory,motor,interneuron,poly
    ind_type[0] = np.where(
        np.logical_and(np.logical_and(issensory.astype(bool), (1 - ismotor).astype(bool)),
                       (1 - isinterneuron).astype(bool)))[0]
    ind_type[1] = np.where(
        np.logical_and(np.logical_and((1 - issensory).astype(bool), ismotor.astype(bool)),
                       (1 - isinterneuron).astype(bool)))[0]
    ind_type[2] = np.where(
        np.logical_and(np.logical_and((1 - issensory).astype(bool), (1 - ismotor).astype(bool)),
                       isinterneuron.astype(bool)))[0]

    ind_type[3] = np.where(issensory + ismotor + isinterneuron >= 2)[0]

    # Head, Middle, Tail

    ind_pos = [[] for _ in range(3)]
    ind_pos[0] = [i for i, j in enumerate(location) if j == 'H']
    ind_pos[1] = [i for i, j in enumerate(location) if j == 'M']
    ind_pos[2] = [i for i, j in enumerate(location) if j == 'T']

    ind_type_pos_number = np.zeros((3, 4))

    ind_type_pos = [[] for _ in range(3)]

    for j in range(3):
        ind_type_pos[j] = [[] for _ in range(4)]

    for i in range(4):
        for j in range(3):
            ind_type_pos[j][i] = [val for val in ind_pos[j] if val in ind_type[i]]
            ind_type_pos_number[j, i] = len(ind_type_pos[j][i])

    ind_neuron_subsampled = [[] for _ in range(3) for _ in range(4)]
    for j in range(3):
        ind_neuron_subsampled[j] = [[] for _ in range(4)]

    for i in range(4):
        for j in range(3):
            try:
                ind_neuron_subsampled[j][i] = np.random.choice(ind_type_pos[j][i],
                                                               np.floor(ind_type_pos_number[j, i] * props[j, i]).astype(
                                                                   int), replace=False)
            except:
                ind_neuron_subsampled[j][i] = []

    ind_neuron_subsampled = np.sort(
        np.concatenate([np.concatenate(ind_neuron_subsampled[j][:], axis=0) for j in range(3)]).astype(int))

    As = As[np.ix_(range(3), ind_neuron_subsampled, ind_neuron_subsampled)]
    xpos = np.array(deepcopy(xpos[ind_neuron_subsampled]).astype(float))
    names = [j for j0, j in enumerate(names) if j0 in ind_neuron_subsampled]

    return As, names, xpos


# def simulate_celegans(A, posx, M, T, num_given, dthresh,
#                       sigmasq_W, etasq):
#
#     N = A.shape[0]
#     rho = np.mean(A.sum(0))
#
#     # Set sigmasq_W for stability
#     sigmasq_W = sigmasq_W if sigmasq_W is not None else 1./(1.1 * N * rho)
#     W = np.sqrt(sigmasq_W) * npr.randn(N, N)
#
#     W = (npr.randn(N, N) * A)
#     #W =np.identity(N) * A
#     eigmax = np.max(abs(np.linalg.eig(W)[0]))
#
#
#     W = W/ (1.5 * eigmax)
#
#     assert np.all(abs(np.linalg.eigvals(A * W)) <= 1.0)
#
#     # Make a global constraint matrix based on x-position
#     C = np.eye(N, dtype=bool)
#
#     dpos = abs(posx[:,None] - posx[None, :])
#     C[dpos < dthresh] = True
#
#     # Sample permutations for each worm
#     perms = []
#     Ps = np.zeros((M, N, N))
#     for m in range(M):
#         # perm[i] = index of neuron i in worm m's neurons
#         perm = npr.permutation(N)
#         perms.append(perm)
#         Ps[m, np.arange(N), perm] = 1
#         #Ps[m,:,:] = np.identity(N)
#
#     # Make constraint matrices for each worm
#     Cs = np.zeros((M, N, N), dtype=bool)
#     for m, (Cm, Pm, permm) in enumerate(zip(Cs, Ps, perms)):
#         # C is in canonical x canonical
#         # make it canonical x worm[m] order
#         Cm = C.dot(Pm)
#
#         # Randomly choose a handful of given neurons
#         given = npr.choice(N, replace=False, size=num_given)
#         Cm[given, :] = 0
#         Cm[:,permm[given]] = 0
#         Cm[given,permm[given]] = 1
#         Cs[m] = Cm
#         assert np.sum(Pm * Cm) == N
#
#         # plt.figure()
#         # plt.subplot(131)
#         # plt.imshow(Ps[m], interpolation="none")
#         #
#         # plt.subplot(132)
#         # plt.imshow(Cm, interpolation="none")
#         #
#         # plt.subplot(133)
#         # plt.imshow(Ps[m] * Cm, interpolation="none")
#         #
#         # plt.show()
#
#     # Sample some data!
#     Ys = np.zeros((M, T, N))
#     for m in range(M):
#         Ys[m,0,:] = np.zeros(N)
#         Wm = Ps[m].T.dot((W*A).dot(Ps[m]))
#         #Wm = Ps[m].dot((W * A).dot(Ps[m]).T)
#         for t in range(1, T):
#             mu_mt = np.dot(Wm, Ys[m, t-1, :])
#
#             Ys[m,t,:] = mu_mt + np.sqrt(etasq) * npr.randn(N)
#         #plt.plot(Ys[m,:,:])
#
#         #plt.show()
#     return Ys, A, W, Ps, Cs


def simulate_celegans(A, posx, M, T, num_given, dthresh=0.01,
                      sigmasq_W=None, etasq=0.1, spectral_factor = 2):
    N = A.shape[0]
    rho = np.mean(A.sum(0))

    # Set sigmasq_W for stability
    sigmasq_W = sigmasq_W if sigmasq_W is not None else 1./(1.1 * N * rho)


    W = (sigmasq_W * npr.randn(N, N) *A)
    #W =np.identity(N) * A
    eigmax = np.max(abs(np.linalg.eig(W)[0]))


    W = W/ (spectral_factor * eigmax)

    assert np.all(abs(np.linalg.eigvals(A * W)) <= 1.0)

    # Make a global constraint matrix based on x-position

    if type(dthresh) is not str:
        C = np.eye(N, dtype=bool)
        dpos = abs(posx[:,None] - posx[None, :])
        C[dpos < dthresh] = True
    else:
        C = np.ones((N, N), dtype = bool)
    # Sample permutations for each worm
    perms = []
    Ps = np.zeros((M, N, N))
    for m in range(M):
        # perm[i] = index of neuron i in worm m's neurons
        perm = npr.permutation(N)
        perms.append(perm)
        Ps[m, np.arange(N), perm] = 1
        #Ps[m, np.arange(N),np.arange(N)] = 1
    # Make constraint matrices for each worm
    Cs = np.zeros((M, N, N), dtype=bool)

    for m, (Cm, Pm, permm) in enumerate(zip(Cs, Ps, perms)):
        # C is in canonical x canonical
        # make it canonical x worm[m] order
        Cm = C.dot(Pm)

        # Randomly choose a handful of given neurons
        given = npr.choice(N, replace=False, size=num_given)
        Cm[given, :] = 0
        Cm[:,permm[given]] = 0
        Cm[given,permm[given]] = 1
        Cs[m] = Cm
        assert np.sum(Pm * Cm) == N

        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(Ps[m], interpolation="none")
        #
        # plt.subplot(132)
        # plt.imshow(Cm, interpolation="none")
        #
        # plt.subplot(133)
        # plt.imshow(Ps[m] * Cm, interpolation="none")
        #
        # plt.show()

    # Sample some data!
    Ys = np.zeros((M, T, N))
    for m in range(M):
        Ys[m,0,:] = np.ones(N)
        Wm = Ps[m].T.dot((W * A).dot(Ps[m]))
        for t in range(1, T):
            mu_mt = np.dot(Wm, Ys[m, t-1, :])
            Ys[m,t,:] = mu_mt + np.sqrt(etasq) * npr.randn(N)

    return Ys, A, W, Ps, Cs



def log_likelihood_single_worm(Y, A, W, P, etasq):
    N = A.shape[0]
    T = Y.shape[0]
    assert Y.shape == (T, N)
    assert A.shape == (N, N)
    assert W.shape == (N, N)
    assert P.shape == (N, N)

    Weff = np.dot(P.T, np.dot(W * A, P))
    Yerr = Y[1:] - np.dot(Y[:-1], Weff.T)
    ll = -0.5 * N * (T - 1) * np.log(2 * np.pi)
    ll += -0.5 * N * (T - 1) * np.log(etasq)
    ll += -0.5 * np.sum(Yerr ** 2 / etasq)
    return ll

def log_likelihood(Ys, A, W, Ps, etasq):
    # Compute log likelihood of observed data given W, Ps
    M = Ps.shape[0]
    ll = 0
    for m in range(M):
        ll += log_likelihood_single_worm(Ys[m], A, W, Ps[m], etasq)
    return ll

### Baselines

### Iterative MAP Estimate Baseline
def run_iterative_map(Ys, A, Cs, etasq, sigmasq_W, W_true, Ps_true,
                      W_init=None, do_update_W=True, max_iter=20):
    # Iterate between solving for W | Ps and Ps | W
    M, T, N = Ys.shape
    assert A.shape == (N, N)

    W = W_init if W_init is not None else np.sqrt(sigmasq_W) * npr.randn(N,N)
    Ps = np.array([perm_to_P(npr.permutation(N)) for _ in range(M)])

    # W | Ps is just a linear regression
    #    y_{mtn} ~ Pm.T (w_n * a_n) Pm y_{m,t-1,:} + eta^2 I
    # Pm y_{mtn} ~ w_n * a_n Pm y_{m,t-1,:} + eta^2 I
    # x_{mtn} ~ w_n[a_n] x_{m,t-1,n}[a_n] + eta^2 I

    def _update_W(Ys, A, Ps, etasq):
        # Collect covariates
        Xs = []
        for Y, P in zip(Ys, Ps):
            Xs.append(np.dot(Y, P.T))
        X = np.vstack(Xs)

        W = np.zeros((N, N))
        for n in range(N):
            xn = X[1:,n]
            Xpn = X[:-1][:,A[n]]
            W[n, A[n]] = np.linalg.solve(
                np.dot(Xpn.T, Xpn) / etasq + sigmasq_W * np.eye(A[n].sum()),
                np.dot(Xpn.T, xn) / etasq)
        return W

    # Pm | W should is a quadratic assignment problem
    # Let y = Ym[1:] and x = Ym[:-1]
    #   (y - P.T W P x)^2
    # = -2 y P.T W P x + x.T P.T W.T P P.T W P x
    # = -2 y.T P.T W P x + x.T P.T W.T W P x
    # = -2 Tr(x y.T P.T W P) + Tr(x x.T P.T W.T W P)
    def _update_Pm(Ym, A, W, Cm):
        yp, y = Ym[:-1], Ym[1:]
        A1 = -2 * np.sum(yp[:, None, :] * y[:, :, None], axis=0)
        B1 = (W * A)
        A2 = np.sum(yp[:, None, :] * yp[:, :, None], axis=0)
        B2 = np.dot((W * A).T, (W * A))

        # todo: incorporate constraint matrix Cm
        C1 = np.zeros((N, N))
        C2 = 1e8 * (1-Cm)

        Pm = solve_qap(np.array([A1, A2]),
                       np.array([B1, B2]),
                       np.array([C1, C2]))

        # Note that solve_qap uses actually yields the transpose of P
        return Pm.T

    lls = []
    mses = []
    num_corrects = []
    W_samples = []
    Ps_samples = []
    times = []
    def collect_stats(W, Ps):
        times.append(time.time())
        lls.append(log_likelihood(Ys, A, W, Ps, etasq) / (M * T * N))
        W_samples.append(W)
        Ps_samples.append(Ps)
        mses.append(np.mean((W * A - W_true * A) ** 2))

        # Round doubly stochastic matrix P to the nearest permutation matrix
        num_correct = np.zeros(M)
        for m, P in enumerate(Ps):
            row, col = linear_sum_assignment(-P + 1e8 * (1 - Cs[m]))
            num_correct[m] = n_correct(perm_to_P(col), Ps_true[m])
        num_corrects.append(num_correct)

    def callback(W, Ps, t):
        collect_stats(W, Ps)
        print("Iteration {}.  LL: {:.4f}  MSE(W): {:.4f}  Num Correct: {}"
              .format(t, lls[-1], mses[-1], num_corrects[-1]))

    # Run the iterative solver
    ll_prev = log_likelihood(Ys, A, W, Ps, etasq)
    callback(W, Ps, -1)
    for itr in range(max_iter):
        W = _update_W(Ys, A, Ps, etasq)
        for m in range(M):
            Ps[m] = _update_Pm(Ys[m], A, W, Cs[m])
        callback(W, Ps, itr)

        # Check for convergence
        ll_new = log_likelihood(Ys, A, W, Ps, etasq)
        if abs(ll_new - ll_prev) / (N * T * M) < 1e-4:
            break
        ll_prev = ll_new

    times = np.array(times)
    times -= times[0]
    return times, np.array(lls), np.array(mses), \
           np.array(num_corrects), np.array(W_samples), \
           np.array(Ps_samples)


### MCMC
def run_naive_mcmc(Ys, A, Cs, etasq, sigmasq_W, W_true, Ps_true,
                   num_iters=500, num_mh_per_iter=1000,
                   W_init=None, Ps_init=None, do_update_W=True):
    # Iterate between solving for W | Ps and Ps | W
    M, T, N = Ys.shape
    assert A.shape == (N, N)

    # W = np.sqrt(sigmasq_W) * npr.randn(N, N)
    W = W_init if W_init is not None else np.sqrt(sigmasq_W) * npr.randn(N, N)

    # Initialize permutations and ensure they are valid
    Ps = Ps_init if Ps_init is not None else \
        np.array([perm_to_P(npr.permutation(N)) for _ in range(M)])
    for m, (P, C) in enumerate(zip(Ps, Cs)):
        P = round_to_perm( P - 1e8 * (1-C))
        assert np.sum(P[C]) == N
        Ps[m] = P

    # assert False

    # W | Ps is just a linear regression
    #    y_{mtn} ~ Pm.T (w_n * a_n) Pm y_{m,t-1,:} + eta^2 I
    # Pm y_{mtn} ~ w_n * a_n Pm y_{m,t-1,:} + eta^2 I
    # x_{mtn} ~ w_n[a_n] x_{m,t-1,n}[a_n] + eta^2 I
    def _update_W(Ys, A, Ps, etasq):
        # Collect covariates
        Xs = []
        for Y, P in zip(Ys, Ps):
            Xs.append(np.dot(Y, P.T))
        X = np.vstack(Xs)

        W = np.zeros((N, N))
        for n in range(N):
            if np.sum(A[n]) == 0:
                continue

            xn = X[1:, n]
            Xpn = X[:-1][:, A[n]]
            Jn = np.dot(Xpn.T, Xpn) / etasq + sigmasq_W * np.eye(A[n].sum())
            Sign = np.linalg.inv(Jn)
            hn = np.dot(Xpn.T, xn) / etasq
            W[n, A[n]] = npr.multivariate_normal(np.dot(Sign, hn), Sign)
        return W

    # Identify the uncertain rows ahead of time
    def _naive_mh_step(Pm, Ym, A, W, Cm, curr_ll=None):
        # Randomly choose two neurons to swap
        unknowns = np.where(Cm.sum(axis=1) > 1)[0]
        n1, n2 = npr.choice(unknowns, 2, replace=False)
        v1 = np.where(Pm[n1])[0][0]
        v2 = np.where(Pm[n2])[0][0]
        if not Cm[n1, v2] or not Cm[n2, v1]:
            return Pm, curr_ll

        # Forward and Backward proposal probabilities are the same
        # so we just need to evaluate the log likelihoods
        curr_ll = curr_ll if curr_ll is not None else \
            log_likelihood_single_worm(Ym, A, W, Pm, etasq)

        P_prop = Pm.copy()
        P_prop[n1] = Pm[n2]
        P_prop[n2] = Pm[n1]
        prop_ll = log_likelihood_single_worm(Ym, A, W, P_prop, etasq)

        # Randomly accept or reject
        if np.log(npr.rand()) < prop_ll - curr_ll:
            return P_prop, prop_ll
        else:
            return Pm.copy(), curr_ll

    # Sample Pm | W with Metropolis Hastings
    def _update_Pm(Ym, A, W, Cm):
        Pm = Ps[m]
        curr_ll = None
        for _ in range(num_mh_per_iter):
            Pm, curr_ll = _naive_mh_step(Pm, Ym, A, W, Cm, curr_ll=curr_ll)
            # Pm, curr_ll = _smart_mh_step(Pm, Ym, A, W, Cm, curr_ll=curr_ll)

            # Check validity
            assert Pm[Cm].sum() == N
        return Pm

    lls = []
    mses = []
    num_corrects = []
    W_samples = []
    Ps_samples = []
    times = []

    def collect_stats(W, Ps):
        times.append(time.time())
        lls.append(log_likelihood(Ys, A, W, Ps, etasq) / (M * T * N))
        W_samples.append(W)
        Ps_samples.append(Ps)
        mses.append(np.mean((W * A - W_true * A) ** 2))

        # Round doubly stochastic matrix P to the nearest permutation matrix
        num_correct = np.zeros(M)
        for m, P in enumerate(Ps):
            row, col = linear_sum_assignment(-P + 1e8 * (1 - Cs[m]))
            num_correct[m] = n_correct(perm_to_P(col), Ps_true[m])
        num_corrects.append(num_correct)


    def callback(W, Ps, t):
        collect_stats(W, Ps)
        print("MCMC Iteration {}.  LL: {:.4f}  MSE(W): {:.4f}  Num Correct: {}"
              .format(t, lls[-1], mses[-1], num_corrects[-1]))

    # Run the MCMC algorithm
    callback(W, Ps, -1)
    for itr in range(num_iters):
        # Resample weights
        if do_update_W:
            W = _update_W(Ys, A, Ps, etasq)
        # Resample permutations
        for m in range(M):
            Ps[m] = _update_Pm(Ys[m], A, W, Cs[m])
        callback(W, Ps, itr)

    times = np.array(times)
    times -= times[0]



    return times, np.array(lls), np.array(mses), \
           np.array(num_corrects)



def run_smart_mcmc(Ys, A, Cs, etasq, sigmasq_W, W_true, Ps_true,
                   num_iters=200, num_mh_per_iter=1000,
                   W_init=None, Ps_init=None, do_update_W=True):
    # Iterate between solving for W | Ps and Ps | W
    M, T, N = Ys.shape
    assert A.shape == (N, N)

    # W = np.sqrt(sigmasq_W) * npr.randn(N, N)
    W = W_init if W_init is not None else np.sqrt(sigmasq_W) * npr.randn(N, N)

    # Initialize permutations and ensure they are valid
    Ps = Ps_init if Ps_init is not None else \
        np.array([perm_to_P(npr.permutation(N)) for _ in range(M)])
    for m, (P, C) in enumerate(zip(Ps, Cs)):
        P = round_to_perm( P - 1e8 * (1-C))
        assert np.sum(P[C]) == N
        Ps[m] = P

    # W | Ps is just a linear regression
    #    y_{mtn} ~ Pm.T (w_n * a_n) Pm y_{m,t-1,:} + eta^2 I
    # Pm y_{mtn} ~ w_n * a_n Pm y_{m,t-1,:} + eta^2 I
    # x_{mtn} ~ w_n[a_n] x_{m,t-1,n}[a_n] + eta^2 I
    def _update_W(Ys, A, Ps, etasq):
        # Collect covariates
        Xs = []
        for Y, P in zip(Ys, Ps):
            Xs.append(np.dot(Y, P.T))
        X = np.vstack(Xs)

        W = np.zeros((N, N))
        for n in range(N):
            xn = X[1:, n]
            Xpn = X[:-1][:, A[n]]
            Jn = np.dot(Xpn.T, Xpn) / etasq + sigmasq_W * np.eye(A[n].sum())
            Sign = np.linalg.inv(Jn)
            hn = np.dot(Xpn.T, xn) / etasq
            W[n, A[n]] = npr.multivariate_normal(np.dot(Sign, hn), Sign)
        return W

    # Identify the uncertain rows ahead of time
    def _get_valid_swaps(Pm, Cm, n1):
        import ipdb; ipdb.set_trace()
        # Find rows such that when we swap them, the constraints are satisfied
        # Get current assignment of n1
        v1 = np.where(Pm[n1])[0][0]
        # Find rows where Cm[:,v1] = True; these are potential swaps
        poss = np.where(Cm[:,v1])[0]
        # Find current assignments for each of those rows
        vs_poss = np.where(Pm[poss])[1]
        # Of these, which are valid for n1
        valid = np.where(Cm[n1,vs_poss])[0]
        return poss[valid]

    # todo: finish this up... the forward and backward proposal
    # todo: probabilties are a bit tricky!
    def _update_Pm(Pm, Ym, A, W, Cm, curr_ll=None):
        # Identify possible partners to swap with
        n1 = npr.choice(N)
        poss = _get_valid_swaps(Pm, Cm, n1)
        n2 = npr.choice(poss)

        # DEBUG
        v1 = np.where(Pm[n1])[0][0]
        v2 = np.where(Pm[n2])[0][0]
        assert Cm[n1, v2] and Cm[n2, v1]

        if n1 == n2:
            return Pm, curr_ll

        P_prop = Pm.copy()
        P_prop[n1] = Pm[n2]
        P_prop[n2] = Pm[n1]

        # Now forward and backward probabilities differ
        log_p_fwd = -np.log(poss.size)

        # This is tricky: the backward probability involves selecting
        # either n1 and then n2, or n2 then n1.  But now both are measured
        # according to
        bwd_poss = _get_valid_swaps(P_prop, Cm, )
        raise NotImplementedError

        # so we just need to evaluate the log likelihoods
        curr_ll = curr_ll if curr_ll is not None else \
            log_likelihood_single_worm(Ym, A, W, Pm, etasq)

        prop_ll = log_likelihood_single_worm(Ym, A, W, P_prop, etasq)

        # Randomly accept or reject
        if np.log(npr.rand()) < prop_ll - curr_ll:
            return P_prop, prop_ll
        else:
            return Pm.copy(), curr_ll


    lls = []
    mses = []
    num_corrects = []
    W_samples = []
    Ps_samples = []
    times = []

    def collect_stats(W, Ps):
        times.append(time.time())
        lls.append(log_likelihood(Ys, A, W, Ps, etasq) / (M * T * N))
        W_samples.append(W)
        Ps_samples.append(Ps)
        mses.append(np.mean((W * A - W_true * A) ** 2))

        # Round doubly stochastic matrix P to the nearest permutation matrix
        num_correct = np.zeros(M)
        for m, P in enumerate(Ps):
            row, col = linear_sum_assignment(-P + 1e8 * (1 - Cs[m]))
            num_correct[m] = n_correct(perm_to_P(col), Ps_true[m])
        num_corrects.append(num_correct)

    def callback(W, Ps, t):
        collect_stats(W, Ps)
        print("Iteration {}.  LL: {:.4f}  MSE(W): {:.4f}  Num Correct: {}"
              .format(t, lls[-1], mses[-1], num_corrects[-1]))

    # Run the MCMC algorithm
    for itr in range(num_iters):
        # Resample weights
        if do_update_W:
            W = _update_W(Ys, A, Ps, etasq)
        # Resample permutations
        for m in range(M):
            Ps[m] = _update_Pm(Ys[m], A, W, Cs[m])
        callback(W, Ps, itr)

    times = np.array(times)
    times -= times[0]
    return times, np.array(lls), np.array(mses), \
           np.array(num_corrects), np.array(W_samples), \
           np.array(Ps_samples)



### Variational inference

# Helpers to convert params into a random permutation-ish matrix
def perm_to_P(perm):
    K = len(perm)
    P = np.zeros((K, K))
    P[np.arange(K), perm] = 1
    return P

def round_to_perm(P):
    N = P.shape[0]
    assert P.shape == (N, N)
    row, col = linear_sum_assignment(-P)
    P = np.zeros((N, N))
    P[row, col] = 1.0
    return P

def n_correct(P1,P2):
    return P1.shape[0] - np.sum(np.abs(P1-P2))/2.0

def sinkhorn_logspace(logP, niters=10):
    for _ in range(niters):
        # Normalize columns and take the log again
        logP = logP - logsumexp(logP, axis=0, keepdims=True)
        # Normalize rows and take the log again
        logP = logP - logsumexp(logP, axis=1, keepdims=True)
    return logP

def make_map(C):
    assert C.dtype == bool and C.ndim == 2
    N1, N2 = C.shape
    valid_inds = np.where(np.ravel(C))[0]
    C_map = np.zeros((N1 * N2, C.sum()))
    C_map[valid_inds, np.arange(C.sum())] = 1

    def unpack_vec(v):
        return np.reshape(np.dot(C_map, v), (N1, N2))

    def pack_matrix(A):
        return A[C]

    return unpack_vec, pack_matrix

def initialize_params(A, Cs, map_W=None, map_Ps=None):
    N = A.shape[0]
    assert A.shape == (N, N)
    M = Cs.shape[0]
    assert Cs.shape == (M, N, N)

    unpack_W, pack_W = make_map(A)
    mu_W = np.zeros(A.sum()) if map_W is None else pack_W(map_W)
    log_sigmasq_W = -10 * np.ones(A.sum())

    log_mu_Ps = []
    log_sigmasq_Ps = []
    unpack_Ps = []
    for i,C in enumerate(Cs):
        unpack_P, pack_P = make_map(C)
        unpack_Ps.append(unpack_P)
        log_mu_Ps.append(
            np.zeros(C.sum()) if map_Ps is None else np.log(pack_P(map_Ps[i])+1e-8))
        log_sigmasq_Ps.append(-2 * np.ones(C.sum()))

    return mu_W, log_sigmasq_W, unpack_W, \
           log_mu_Ps, log_sigmasq_Ps, unpack_Ps



def q_entropy(log_sigmasq_P, temp):
    return gaussian_entropy(0.5 * log_sigmasq_P) + log_sigmasq_P.size * np.log(temp)


def run_variational_inference(Ys, A, W_true, Ps_true, Cs, etasq,stepsize = 0.1,
                              init_with_true=True, num_iters=250, sigmasq_P =0.1, num_sinkhorn =10,
                              num_mcmc_samples=500, sigma_Lim = [0,0.005], temp=1):

    def sample_q(params, unpack_W, unpack_Ps, Cs, num_sinkhorn, sigma_Lim, temp):

        # Sample W
        mu_W, log_sigmasq_W, log_mu_Ps, log_sigmasq_Ps = params
        W_flat = mu_W + np.sqrt(np.exp(log_sigmasq_W)) * npr.randn(*mu_W.shape)

        W = unpack_W(W_flat)

        # Sample Ps: run sinkhorn to move mu close to Birkhoff
        Ps = []
        for log_mu_P, log_sigmasq_P, unpack_P, C in \
                zip(log_mu_Ps, log_sigmasq_Ps, unpack_Ps, Cs):
            # Unpack the mean, run sinkhorn, the pack it again
            log_mu_P = unpack_P(log_mu_P)
            log_mu_P = sinkhorn_logspace(log_mu_P - 1e8 * (1 - C), num_sinkhorn)

            log_mu_P = log_mu_P[C]

            log_sigmasq_P = log_sigmasq_P

            ##Notice how we limit the variance
            P = np.exp(log_mu_P) + \
                np.sqrt(logistic(log_sigmasq_P) * (sigma_Lim[1] - sigma_Lim[0]) + sigma_Lim[0]) * \
                npr.randn(*log_mu_P.shape)
            P = unpack_P(P)
            # Round to nearest permutation

            if (temp < 1):
                Phat = round_to_perm(P if isinstance(P, np.ndarray) else P.value)
                P = P * temp + (1 - temp) * Phat

            # Ps.append(np.identity(P.shape[0]))

            Ps.append(P)

        Ps = np.array(Ps)
        return W, Ps

    def elbo(params, unpack_W, unpack_Ps, Ys, A, Cs, etasq, sigmasq_P,
             num_sinkhorn, num_mcmc_samples, sigma_Lim, temp):
        """
        Provides a stochastic estimate of the variational lower bound.
        sigma_Lim: limits for the variance of the re-parameterization of the permutation
        """

        def unconstrained_log_prior(P, sigmasq_P):
            """
            Consider a product (coordinate-wise) of mixtures of
            two gaussians with std sigma_prior and centers at 0 and 1)
            """
            N = P.shape[0]
            assert P.shape == (N, N)
            corners = np.array([0, 1])
            diffs = P[:, :, None] - corners[None, None, :]
            return np.sum(logsumexp(-0.5 * diffs ** 2 / sigmasq_P, axis=2)) \
                   - 0.5 * N ** 2 * np.log(2 * np.pi) \
                   - 0.5 * N ** 2 * np.log(sigmasq_P)

        M, T, N = Ys.shape
        assert A.shape == (N, N)
        assert len(unpack_Ps) == M

        mu_W, log_sigmasq_W, log_mu_Ps, log_sigmasq_Ps = params

        L = 0

        for smpl in range(num_mcmc_samples):
            W, Ps = sample_q(params, unpack_W, unpack_Ps, Cs, num_sinkhorn, sigma_Lim, temp)

            # Compute the ELBO
            L += log_likelihood(Ys, A, W, Ps, etasq) / num_mcmc_samples
            L += np.sum([unconstrained_log_prior(P, sigmasq_P) for P in Ps]) / num_mcmc_samples

        # Add the entropy terms
        L += np.sum([q_entropy(np.log(logistic(log_sigmasq_P) * (sigma_Lim[1] - sigma_Lim[0]) + sigma_Lim[0]), temp) for
                     log_sigmasq_P in log_sigmasq_Ps])

        L += gaussian_entropy(0.5 * log_sigmasq_W)

        ## This latter term was missing, for details see the appendix of the VAE paper
        L += - 0.5 * log_sigmasq_W.size * (np.log(2 * np.pi)) - 0.5 * np.sum(np.exp(log_sigmasq_W)) - 0.5 * np.sum(
            np.power(mu_W, 2))
        # Normalize objective

        L /= (T * M * N)

        return L

    M, T, N = Ys.shape
    # Initialize variational parameters
    if init_with_true:
        mu_W, log_sigmasq_W, unpack_W, log_mu_Ps, log_sigmasq_Ps, unpack_Ps = \
            initialize_params(A, Cs, map_Ps=Ps_true, map_W=W_true)
    else:
        mu_W, log_sigmasq_W, unpack_W, log_mu_Ps, log_sigmasq_Ps, unpack_Ps = \
            initialize_params(A, Cs)

    # Make a function to convert an array of params into
    # a set of parameters mu_W, sigmasq_W, [mu_P1, sigmasq_P1, ... ]
    flat_params, unflatten = \
        flatten((mu_W, log_sigmasq_W, log_mu_Ps, log_sigmasq_Ps))

    objective = \
        lambda flat_params, t: \
            -1 * elbo(unflatten(flat_params), unpack_W, unpack_Ps, Ys, A, Cs, etasq,
                      sigmasq_P, num_sinkhorn, num_mcmc_samples, sigma_Lim, temp)

    # Define a callback to monitor optimization progress
    elbos = []
    lls=[]
    mses = []

    num_corrects = []
    times = []

    def collect_stats(params):
        times.append(time.time())
        elbos.append(-1 * objective(params, 0))

        # Sample the variational posterior and compute num correct matches
        mu_W, log_sigmasq_W, log_mu_Ps, log_sigmasq_Ps = unflatten(params)

        W, Ps = sample_q(unflatten(params), unpack_W, unpack_Ps, Cs, 10, sigma_Lim, 0)

        lls.append(log_likelihood(Ys, A, W, Ps, etasq) / (M * T * N))


        list=[]
        for i in range(A.shape[0]):
            list.extend(np.where(Ps[0, i, :]+Ps_true[0, i, :] ==1)[0])

        mses.append(np.mean((W * A - W_true * A) ** 2))


        # Round doubly stochastic matrix P to the nearest permutation matrix
        num_correct = np.zeros(M)
        for m, P in enumerate(Ps):
            row, col = linear_sum_assignment(-P + 1e8 * (1 - Cs[m]))
            num_correct[m] = n_correct(perm_to_P(col), Ps_true[m])
        num_corrects.append(num_correct)

    def callback(params, t, g):
        collect_stats(params)
        print("Iteration {}.  ELBO: {:.4f} LL: {:.4f} MSE(W): {:.4f}, Num Correct: {}"
              .format(t, elbos[-1], lls[-1], mses[-1], num_corrects[-1]))

    # Run optimizer

    callback(flat_params, -1, None)
    variational_params = adam(grad(objective),
                              flat_params,
                              step_size=stepsize,
                              num_iters=num_iters,
                              callback=callback)

    times = np.array(times)
    times -= times[0]


    W_samples = []
    Ps_samples = []

    for i in range(100):
        W, Ps = sample_q(unflatten(variational_params[0]), unpack_W, unpack_Ps, Cs, num_sinkhorn, sigma_Lim, temp)
        W_samples.append(W)
        Ps_samples.append(Ps)
    print np.array(Ps_samples).shape
    return times, np.array(elbos), np.array(lls), np.array(mses), \
           np.array(num_corrects), np.array(W_samples), \
           np.array(Ps_samples)


def run_naive_variational_inference(Ys, A, W_true, Ps_true, Cs, etasq,stepsize = 0.1,
                              init_with_true=True, num_iters=250, sigmasq_P =0.1, num_sinkhorn =10,
                              num_mcmc_samples=500, sigma_Lim = [0,0.005], temp=1):

    def sample_q(params, unpack_W, unpack_Ps, Cs, num_sinkhorn, sigma_Lim, temp):

        # Sample W
        mu_W, log_sigmasq_W, log_mu_Ps, log_sigmasq_Ps = params
        W_flat = mu_W + np.sqrt(np.exp(log_sigmasq_W)) * npr.randn(*mu_W.shape)

        W = unpack_W(W_flat)

        Ps = []
        for log_mu_P, log_sigmasq_P, unpack_P, C in \
                zip(log_mu_Ps, log_sigmasq_Ps, unpack_Ps, Cs):
            # Unpack the mean, run sinkhorn, the pack it again
            log_mu_P = unpack_P(log_mu_P)
            log_mu_P = log_mu_P - logsumexp(log_mu_P)
            log_mu_P = log_mu_P[C]

            log_sigmasq_P = log_sigmasq_P

            ##Notice how we limit the variance
            P = np.exp(log_mu_P) + \
                np.sqrt(logistic(log_sigmasq_P) * (sigma_Lim[1] - sigma_Lim[0]) + sigma_Lim[0]) * \
                npr.randn(*log_mu_P.shape)

            P = unpack_P(P)

            # Round each row to the closest index (neuron)
            if (temp < 1):
                ind_one_hot = np.argmax(P, axis = 1)
                Phat = np.zeros((N, N))
                Phat[range(N), ind_one_hot] = 1

                #Phat = round_to_perm(P if isinstance(P, np.ndarray) else P.value)
                P = P * temp + (1 - temp) * Phat


            Ps.append(P)

        Ps = np.array(Ps)
        return W, Ps


    def elbo(params, unpack_W, unpack_Ps, Ys, A, Cs, etasq, sigmasq_P,
             num_sinkhorn, num_mcmc_samples, sigma_Lim, temp):
        """
        Provides a stochastic estimate of the variational lower bound.
        sigma_Lim: limits for the variance of the re-parameterization of the permutation
        """
        sigma_w_prior = 0.000000000001
        def unconstrained_log_prior(P, sigmasq_P):
            """
            Consider a row-wise product of mixtures of M gaussians, with centers at the N N-dimensional one-hot vectors

            """
            M = P.shape[0]
            N = P.shape[1]
            # assert P.shape == (N, N)
            corners = np.eye(N)
            # corners = corners[:,:3]
            diffs = P[:, :, None] - corners[None, :, :]

            ss = -0.5 * np.sum(diffs ** 2, axis=1) / sigmasq_P - N * 0.5 * np.log(2 * np.pi)

            return np.sum(logsumexp(ss, axis=1) - np.log(N))

        M, T, N = Ys.shape
        assert A.shape == (N, N)
        assert len(unpack_Ps) == M

        mu_W, log_sigmasq_W, log_mu_Ps, log_sigmasq_Ps = params

        L = 0

        for smpl in range(num_mcmc_samples):
            W, Ps = sample_q(params, unpack_W, unpack_Ps, Cs, num_sinkhorn, sigma_Lim, temp)

            # Compute the ELBO
            L += log_likelihood(Ys, A, W, Ps, etasq) / num_mcmc_samples
            L += np.sum([unconstrained_log_prior(P, sigmasq_P) for P in Ps]) / num_mcmc_samples

        # Add the entropy terms
        L += np.sum([q_entropy(np.log(logistic(log_sigmasq_P) * (sigma_Lim[1] - sigma_Lim[0]) + sigma_Lim[0]), temp) for
                     log_sigmasq_P in log_sigmasq_Ps])

        L += gaussian_entropy(0.5 * log_sigmasq_W)

        ## This latter term was missing, for details see the appendix of the VAE paper
        L += - 0.5 * log_sigmasq_W.size * (np.log(2 * np.pi *sigma_w_prior))  - 0.5 * np.sum(np.exp(log_sigmasq_W)) /sigma_w_prior - 0.5 * np.sum(np.power(mu_W, 2)) / sigma_w_prior
        # Normalize objective

        L /= (T * M * N)

        return L

    M, T, N = Ys.shape
    # Initialize variational parameters
    if init_with_true:
        mu_W, log_sigmasq_W, unpack_W, log_mu_Ps, log_sigmasq_Ps, unpack_Ps = \
            initialize_params(A, Cs, map_Ps=Ps_true, map_W=W_true)
    else:
        mu_W, log_sigmasq_W, unpack_W, log_mu_Ps, log_sigmasq_Ps, unpack_Ps = \
            initialize_params(A, Cs)

    # Make a function to convert an array of params into
    # a set of parameters mu_W, sigmasq_W, [mu_P1, sigmasq_P1, ... ]
    flat_params, unflatten = \
        flatten((mu_W, log_sigmasq_W, log_mu_Ps, log_sigmasq_Ps))

    objective = \
        lambda flat_params, t: \
            -1 * elbo(unflatten(flat_params), unpack_W, unpack_Ps, Ys, A, Cs, etasq,
                      sigmasq_P, num_sinkhorn, num_mcmc_samples, sigma_Lim, temp)

    # Define a callback to monitor optimization progress
    elbos = []
    lls=[]
    mses = []

    num_corrects = []
    times = []

    def collect_stats(params):
        times.append(time.time())
        elbos.append(-1 * objective(params, 0))

        # Sample the variational posterior and compute num correct matches
        #mu_W, log_sigmasq_W, log_mu_Ps, log_sigmasq_Ps = unflatten(params)

        W, Ps = sample_q(unflatten(params), unpack_W, unpack_Ps, Cs, 10, sigma_Lim, 0)

        lls.append(log_likelihood(Ys, A, W, Ps, etasq) / (M * T * N))


        list=[]
        for i in range(A.shape[0]):
            list.extend(np.where(Ps[0, i, :]+Ps_true[0, i, :] ==1)[0])

        mses.append(np.mean((W * A - W_true * A) ** 2))


        # Round doubly stochastic matrix P to the nearest permutation matrix
        num_correct = np.zeros(M)
        for m, P in enumerate(Ps):
            row, col = linear_sum_assignment(-P + 1e8 * (1 - Cs[m]))
            num_correct[m] = n_correct(perm_to_P(col), Ps_true[m])
        num_corrects.append(num_correct)

    def callback(params, t, g):
        collect_stats(params)
        print("Iteration {}.  ELBO: {:.4f} LL: {:.4f} MSE(W): {:.4f}, Num Correct: {}"
              .format(t, elbos[-1], lls[-1], mses[-1], num_corrects[-1]))

    # Run optimizer

    callback(flat_params, -1, None)
    variational_params = adam(grad(objective),
                              flat_params,
                              step_size=stepsize,
                              num_iters=num_iters,
                              callback=callback)

    times = np.array(times)
    times -= times[0]

    W_samples = []
    Ps_samples = []

    for i in range(100):
        W, Ps = sample_q(unflatten(variational_params[0]), unpack_W, unpack_Ps, Cs, num_sinkhorn, sigma_Lim, temp)
        W_samples.append(W)
        Ps_samples.append(Ps)

    return times, np.array(elbos), np.array(lls), np.array(mses), \
           np.array(num_corrects), np.array(W_samples), \
           np.array(Ps_samples)






def plot_results(experiment_name, results_vi, results_naive_vi, results_mcmc, results_map):
    # Plots:
    #  - Log Likelihood over time
    #  - MSE of W over time
    #  - Num correct over time
    times_vi, elbos_vi, lls_vi, mses_vi, ncs_vi, Ws_vi, Ps_vi = results_vi
    times_naive_vi, elbos_naive_vi, lls_naive_vi, mses_naive_vi, ncs_naive_vi, Ws_naive_vi, Ps_naive_vi = results_naive_vi
    times_mcmc, lls_mcmc, mses_mcmc, ncs_mcmc = results_mcmc
    times_map, lls_map, mses_map, ncs_map, Ws_map, Ps_map = results_map
    t_max = np.max([times_vi[-1], times_mcmc[-1], times_map[-1]])

    fig = plt.figure(figsize=(5.5, 2.5))

    # Panel 1: Expected log likelihood
    # ax0 = fig.add_subplot(131)
    # ax0.plot(times_vi, elbos_vi, color=colors[0])
    # ax0.set_xlabel("Time (s)")
    # ax0.set_xlim(1, t_max)
    # ax0.set_xscale("log")
    # ax0.set_ylabel("ELBO")

    ax0 = fig.add_subplot(131)
    ax0.plot(times_vi, lls_vi, color=colors[0])
    ax0.plot(times_naive_vi, lls_naive_vi, color=colors[3])
    ax0.plot(times_mcmc, lls_mcmc, color=colors[1])
    ax0.plot(times_map, lls_map, color=colors[2])

    ax0.set_xlabel("Time (s)")
    ax0.set_xlim(1, t_max)
    ax0.set_xscale("log")
    ax0.set_ylabel("log likelihood")


    # Panel 2: MSE(W)
    ax1 = fig.add_subplot(132)
    ax1.plot(times_vi, mses_vi, color=colors[0])
    ax1.plot(times_naive_vi, mses_naive_vi, color=colors[3])
    ax1.plot(times_mcmc, mses_mcmc, color=colors[1])
    ax1.plot(times_map, mses_map, color=colors[2])
    ax1.set_xlabel("Time (s)")
    ax1.set_xlim(1, t_max)
    ax1.set_xscale("log")
    ax1.set_ylabel("MSE($\\mathbf{W}$)")

    N = Ps_vi.shape[2]

    # Panel 3: num correct
    ax2 = fig.add_subplot(133)

    ax2.plot(times_vi, ncs_vi.mean(1)/ N, color=colors[0], label="VI")
    ax2.plot(times_naive_vi, ncs_naive_vi.mean(1)/ N, color=colors[3], label="Naive VI")
    ax2.plot(times_mcmc, ncs_mcmc.mean(1) /N, color=colors[1], label="MCMC")
    ax2.plot(times_map, ncs_map.mean(1) / N, color=colors[2], label="MAP")
    ax2.legend(loc="upper left")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(1, t_max)
    ax2.set_xscale("log")
    ax2.set_ylabel("Avg. Prop. Correct")

    fig.suptitle(experiment_name)
    plt.tight_layout(rect=(0,0,1,0.95))
    plt.savefig(os.path.join(RESULTS_DIR, experiment_name + ".png"))
    plt.savefig(os.path.join(RESULTS_DIR, experiment_name + ".pdf"))





#if __name__ == "__main__":

def run_realistic_experiment():
    Ms = [5]
    Ts = [1000]
    num_given_neuronss = [25]
    dthreshs = [0.005,0.01,0.05,0.1,0.5,1]
    dthreshs =[0.1,0.5,1]
    etasqs = [1]
    rhos =[0.3,0.5,0.7]
    spectral_factors = [2.5]

    # Load the real C Elegans network

    for M, T, num_given_neurons, dthresh, etasq, rho, spectral_factor in \
            it.product(Ms, Ts, num_given_neuronss, dthreshs, etasqs, rhos, spectral_factors):
        experiment_name = "celegans_M{}_T{}_rho{}_giv{}_dthresh{}_etasq{}_sf{}". \
            format(M, T, rho, num_given_neurons, dthresh, etasq, spectral_factor)
        ## chose only a proportion rho of entire connectome
        props = np.ones((3, 4)) * rho
        As, names, xpos, = load_celegans_network(props)

        A = (np.sum(As, axis=0) > 0).astype(bool)
        N = A.shape[0]
        print N

        sigmasq_W = 1. / (1.5 * N * np.mean(A.sum(0)))

        # Simulate a few "worm recordings"
        # sim = cached(RESULTS_DIR, experiment_name + "_data")(simulate_celegans)
        sim = simulate_celegans

        Ys, A, W_true, Ps_true, Cs = \
            sim(A, xpos, M, T, num_given_neurons, dthresh, sigmasq_W, etasq, spectral_factor)

        print("Avg choices: {}".format(Cs.sum(1).mean()))
        print("E[W]: {:.4f}".format(W_true[A].mean()))
        print("Std[W]: {:.4f}".format(W_true[A].std()))

        # Cached VI experiment function
        run_vi = cached(RESULTS_DIR, experiment_name + "_vi")(run_variational_inference)
        results_vi = run_vi(Ys, A, W_true, Ps_true, Cs, etasq, stepsize = 0.1, init_with_true=False, num_iters=200, sigmasq_P =0.1, num_sinkhorn =10, num_mcmc_samples=1, sigma_Lim = [0,0.005], temp=1)

        run_naive_vi = cached(RESULTS_DIR, experiment_name + "_naive_vi")(run_naive_variational_inference)
        results_naive_vi = run_naive_vi(Ys, A, W_true, Ps_true, Cs, etasq, stepsize=0.1, init_with_true=False, num_iters=200, sigmasq_P=0.1, num_sinkhorn=10, num_mcmc_samples=1, sigma_Lim=[0, 0.005], temp=1)

        #Cached MCMC experiment
        sigmasq_W = 1. / (1.5 * N * np.mean(A.sum(0)))
        run_mcmc = cached(RESULTS_DIR, experiment_name + "_mcmc")(run_naive_mcmc)
        results_mcmc = run_mcmc(Ys, A, Cs, etasq, sigmasq_W, W_true, Ps_true, num_iters=2000, W_init=None, do_update_W=True)

        # Cached MAP experiment
        sigmasq_W = 1. / (1.5 * N * np.mean(A.sum(0)))
        run_map = cached(RESULTS_DIR, experiment_name + "_map")(run_iterative_map)
        results_map = run_map(Ys, A, Cs, etasq, sigmasq_W, W_true, Ps_true)

        plot_results(experiment_name, results_vi, results_naive_vi, results_mcmc, results_map)


#if __name__ == "__main__":
    # npr.seed(1)
run_realistic_experiment()
