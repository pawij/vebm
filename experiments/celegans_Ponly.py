import os, sys
from copy import deepcopy

import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import autograd.scipy as scipy
from autograd.scipy.misc import logsumexp
from scipy.optimize import linear_sum_assignment
from autograd.util import flatten
from autograd.core import unbox_if_possible
from autograd.optimizers import adam, sgd
from autograd import grad

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from birkhoff.primitives import gaussian_logp, gaussian_entropy, logistic, logit

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


def simulate_data(M, T, N, num_poss_per_neuron,
                  sigmasq_W=0.08, etasq=0.1, rho=0.1):

    # Sample global worm variables
    A = npr.rand(N, N) < rho
    W = np.sqrt(sigmasq_W) * npr.randn(N, N)
    assert np.all(abs(np.linalg.eigvals(A * W)) <= 1.0)

    # Sample permutations for each worm
    Ps = np.zeros((M, N, N))
    for m in range(M):
        # perm[i] = index of neuron i in worm m's neurons
        perm = npr.permutation(N)
        Ps[m, np.arange(N), perm] = 1

    # Make constraint matrices for each worm
    Cs = np.ones((M, N, N), dtype=bool)
    for m in range(M):
        for n in range(N):
            i = np.where(Ps[m, n, :] == 1)[0]
            invalid = npr.choice(np.arange(N), N - num_poss_per_neuron, replace=False)
            for j in invalid:
                if j != i:
                    Cs[m,n,j] = False

    # Sample some data!
    Ys = np.zeros((M, T, N))
    for m in range(M):
        Ys[m,0,:] = np.ones(N)
        Wm = Ps[m].T.dot((W*A).dot(Ps[m]))
        for t in range(1, T):
            mu_mt = np.dot(Wm, Ys[m, t-1, :])
            Ys[m,t,:] = mu_mt + np.sqrt(etasq) * npr.randn(N)

    return Ys, A, W, Ps, Cs

def log_likelihood(Ys, A, W, Ps, etasq):
    # Compute log likelihood of observed data given W, Ps
    M = Ps.shape[0]
    N = A.shape[0]
    T = Ys.shape[1]
    assert Ys.shape == (M,T,N)
    assert A.shape == (N,N)
    assert W.shape == (N,N)
    assert Ps.shape == (M,N,N)

    ll = 0
    for m in range(M):
        Wm = np.dot(Ps[m].T, np.dot(W * A, Ps[m]))
        Yerr = Ys[m,1:] - np.dot(Ys[m,:-1], Wm.T)
        ll += -0.5 * N * (T-1) * np.log(2 * np.pi)
        ll += -0.5 * N * (T-1) * np.log(etasq)
        ll += -0.5 * np.sum(Yerr**2 / etasq)
    return ll

def unconstrained_log_prior(P, sigmasq_P):
    """
    Consider a product (coordinate-wise) of mixtures of
    two gaussians with std sigma_prior and centers at 0 and 1)
    """
    N = P.shape[0]
    assert P.shape == (N, N)
    corners = np.array([0, 1])
    diffs = P[:,:,None] - corners[None, None, :]
    return np.sum(logsumexp(-0.5 * diffs ** 2 / sigmasq_P, axis=2)) \
           - 0.5 * N**2 * np.log(2 * np.pi) \
           - 0.5 * N**2 * np.log(sigmasq_P)

# Helpers to convert params into a random permutation-ish matrix
def perm_to_P(perm):
    K = len(perm)
    P = np.zeros((K, K))
    P[range(K), perm] = 1
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

def initialize_params(A, Cs, map_Ps=None):
    N = A.shape[0]
    assert A.shape == (N, N)
    M = Cs.shape[0]
    assert Cs.shape == (M, N, N)

    log_mu_Ps = []
    log_sigmasq_Ps = []
    unpack_Ps = []

    for i,C in enumerate(Cs):
        unpack_P, pack_P = make_map(C)
        unpack_Ps.append(unpack_P)
        log_mu_Ps.append(
            np.zeros(C.sum()) if map_Ps is None else np.log(pack_P(map_Ps[i])+1e-8))
        log_sigmasq_Ps.append(-2 * np.ones(C.sum()))

    return log_mu_Ps, log_sigmasq_Ps, unpack_Ps

def sample_q(params, unpack_Ps, Cs, num_sinkhorn, temp=0.1):
    # Sample Ps: run sinkhorn to move mu close to Birkhoff
    log_mu_Ps, log_sigmasq_Ps = params
    Ps = []
    for log_mu_P, log_sigmasq_P, unpack_P, C in \
            zip(log_mu_Ps, log_sigmasq_Ps, unpack_Ps, Cs):

        # Unpack the mean, run sinkhorn, the pack it again
        log_mu_P = unpack_P(log_mu_P)
        log_mu_P = sinkhorn_logspace(log_mu_P - 1e8 * (1-C), num_sinkhorn)
        log_mu_P = log_mu_P[C]

        log_sigmasq_P = log_sigmasq_P
        P = np.exp(log_mu_P) + \
            np.sqrt(np.exp(log_sigmasq_P)) * \
            npr.randn(*log_mu_P.shape)
        P = unpack_P(P)

        # Round to nearest permutation
        Phat = round_to_perm(P if isinstance(P, np.ndarray) else P.value)
        P = P * temp + (1 - temp) * Phat

        Ps.append(P)
    Ps = np.array(Ps)
    return Ps

def q_entropy(log_sigmasq_P, temp):
    return gaussian_entropy(0.5 * log_sigmasq_P) + log_sigmasq_P.size * np.log(temp)

def elbo(params, unpack_Ps, Ys, A, W, Cs, etasq, sigmasq_P,
         num_sinkhorn=5, num_mcmc_samples=1, temp=1.0):
    """
    Provides a stochastic estimate of the variational lower bound.
    """
    M, T, N = Ys.shape
    assert A.shape == (N, N)
    assert len(unpack_Ps) == M

    log_mu_Ps, log_sigmasq_Ps = params

    L = 0
    for smpl in range(num_mcmc_samples):
        Ps = sample_q(params, unpack_Ps, Cs, num_sinkhorn)

        # Compute the ELBO
        L += log_likelihood(Ys, A, W, Ps, etasq) / num_mcmc_samples
        L += np.sum([unconstrained_log_prior(P, sigmasq_P) for P in Ps]) / num_mcmc_samples

    # Add the entropy terms
    L += np.sum([q_entropy(log_sigmasq_P, temp) for log_sigmasq_P in log_sigmasq_Ps])

    # Normalize objective
    L /= (T * M * N)

    return L

if __name__ == "__main__":
    M = 5
    T = 1000
    N = 100
    num_poss_per_neuron = 10
    etasq = 0.1
    Ys, A, W, Ps_true, Cs = simulate_data(M, T, N, num_poss_per_neuron, etasq=etasq)

    # Make sure the true weights have high probability
    print("ll true: {:.4f}".format(log_likelihood(Ys, A, W, Ps_true, etasq) / (M *T * N)))
    print("ll tran: {:.4f}".format(log_likelihood(Ys, A, W.T, Ps_true, etasq) / (M *T * N)))
    print("ll rand: {:.4f}".format(log_likelihood(Ys, A, npr.randn(N,N), Ps_true, etasq) / (M *T * N)))

    # Initialize variational parameters
    log_mu_Ps, log_sigmasq_Ps, unpack_Ps = \
        initialize_params(A, Cs, map_Ps=0.1 + 0.9 * Ps_true)

    # Debug
    # Ps_sample = sample_q((log_mu_Ps, log_sigmasq_Ps), unpack_Ps, Cs, num_sinkhorn=5)
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(Ps_sample[0], interpolation="none", vmin=0, vmax=1)
    # plt.subplot(122)
    # plt.imshow(Ps_true[0], interpolation="none", vmin=0, vmax=1)
    # plt.show()

    # Make a function to convert an array of params into
    # a set of parameters mu_W, sigmasq_W, [mu_P1, sigmasq_P1, ... ]
    flat_params, unflatten = \
        flatten((log_mu_Ps, log_sigmasq_Ps))

    objective = \
        lambda flat_params, t: \
            -1 * elbo(unflatten(flat_params), unpack_Ps, Ys, A, W, Cs, etasq,
                      sigmasq_P=0.1)

    # Define a callback to monitor optimization progress
    elbos = [-1 * objective(flat_params, 0)]
    def callback(flat_params, t, g):
        elbos.append(-1 * objective(flat_params, t))

        # Sample the variational posterior and compute num correct matches
        log_mu_Ps, log_sigmasq_Ps = unflatten(flat_params)
        Ps = sample_q((log_mu_Ps, log_sigmasq_Ps), unpack_Ps, Cs, num_sinkhorn=5)

        # Round doubly stochastic matrix P to the nearest permutation matrix
        num_correct = np.zeros(M)
        for m, P in enumerate(Ps):
            row, col = linear_sum_assignment(-P + 1e8 * (1 - Cs[m]))
            num_correct[m] = n_correct(perm_to_P(col), Ps_true[m])

        print("Iteration {}. ELBO: {:.2f}  Num Correct: {}".
              format(t, elbos[-1], num_correct))

    # Run optimizer
    num_adam_iters = 100
    stepsize = 0.1

    callback(flat_params, -1, None)
    variational_params = adam(grad(objective),
                              flat_params,
                              step_size=stepsize,
                              num_iters=num_adam_iters,
                              callback=callback)
