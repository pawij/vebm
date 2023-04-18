import sys
import signal
import warnings

import matplotlib.pyplot as plt
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

from scipy.optimize import linear_sum_assignment

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam

from birkhoff.primitives import \
    logit, logistic, gaussian_logp, gaussian_entropy, \
    psi_to_birkhoff, log_det_jacobian, birkhoff_to_psi

import scipy as sp

npr.seed(0)

DO_PLOT = True

if __name__ == "__main__":
    # Set up a simple matching problem
    K = 20
    D = 2
    eta = 0.01#0.2
    mus = 2 * npr.randn(K, D)
    num_mcmc_samples = 10
    sigma_min, sigma_max = 1e-3, 5.0

    # Sample a true permutation (in=col, out=row)
    P_true = np.zeros((K, K))
    P_true[np.arange(K), npr.permutation(K)] = 1

    # Sample data according to this permutation
    mus_perm = P_true.dot(mus)
    xs = mus_perm + eta * npr.randn(K, D)

    # Build variational objective.
    # Variational dist is a diagonal Gaussian over the (K-1)**2 parameters
    def unpack_params(params):
        assert params.shape == (2 * (K - 1)**2, )
        mu = np.reshape((params[:(K-1)**2]), (K-1, K-1))
        logit_sigma = np.reshape((params[(K-1)**2:]), (K-1, K-1))
        sigma = sigma_min + (sigma_max - sigma_min) * logistic(logit_sigma)
        log_sigma = np.log(sigma)
        return mu, log_sigma, sigma

    # Set up the log probability objective
    # Assume a uniform prior on P?
    # Right now this is just the likelihood...
    def log_prob(P, t):
        return np.sum(gaussian_logp(xs, np.dot(P, mus), eta))

    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mu, log_sigma, sigma = unpack_params(params)
        Psi_samples = mu + npr.randn(num_mcmc_samples, K-1, K-1) * sigma
        P_samples = [psi_to_birkhoff(logistic(Psi)) for Psi in Psi_samples]

        # Compute ELBO. Explicitly compute gaussian entropy.
        elbo = 0
        for P, Psi in zip(P_samples, Psi_samples):
            elbo = elbo + log_prob(P, t) / num_mcmc_samples
            elbo = elbo - log_det_jacobian(P) / num_mcmc_samples
        elbo = elbo + gaussian_entropy(log_sigma)

        # Minimize the negative elbo
        return -elbo / K

    gradient = grad(variational_objective)
    elbos = []

    ### Plotting
    if DO_PLOT:
        fig = plt.figure(figsize=(8, 4), facecolor='white')
        ax1 = fig.add_subplot(121, frameon=True)
        ax2 = fig.add_subplot(122, frameon=True)
        plt.ion()
        plt.show(block=False)

    def plot_permutation(ax1, ax2, P):
        ax1.imshow(P_true, interpolation="none", vmin=0, vmax=1)
        ax1.set_title("True $\Pi$")
        ax2.imshow(P, interpolation="none", vmin=0, vmax=1)
        ax2.set_title("Inferred $g(\mu)$")


    def callback(params, t, g):
        elbos.append(-variational_objective(params, t))
        print("Iteration {} lower bound {}".format(t, elbos[-1]))

        mu, log_sigma, sigma = unpack_params(params)
        print("sigma min: ", sigma.min(), "\t sigma max: ", sigma.max())

        if DO_PLOT:
            plt.cla()
            Psi = mu + sigma * npr.randn(K - 1, K - 1)
            P = psi_to_birkhoff(logistic(Psi))
            plot_permutation(ax1, ax2, P)
            plt.draw()
            plt.pause(1.0 / 30.0)

        if ctrlc_pressed[0]:
            sys.exit()

    # Check for quit
    ctrlc_pressed = [False]
    def ctrlc_handler(signal, frame):
        print("Halting due to Ctrl-C")
        ctrlc_pressed[0] = True
    signal.signal(signal.SIGINT, ctrlc_handler)

    print("Variational inference for matching...")
    #    init_mean = birkhoff_to_psi(1. / K * np.ones((K - 1, K - 1))).ravel()
    init_mean = birkhoff_to_psi(1. / K * np.ones((K, K))).ravel()
    init_mean = logit(init_mean)
    init_logit_std = -3 * np.ones((K - 1) ** 2)
    init_var_params = np.concatenate([init_mean, init_logit_std])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=100, callback=callback)
    # fig.savefig("permutation_K20.png")

    # Plot the elbo
    plt.figure(figsize=(6,4))
    plt.plot(elbos)
    plt.xlim(0, 100)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.tight_layout()
    plt.savefig("permutation_K20_elbo.png")

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
    
    # Sample from the posterior and show samples
    mu_post, log_sigma_post, sigma_post = unpack_params(variational_params)

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    for i in range(4):
        for j in range(4):
            Psi_sample = mu_post + npr.randn(K - 1, K - 1) * sigma_post
            P_sample = psi_to_birkhoff(logistic(Psi_sample))
            # Round doubly stochastic matrix P to the nearest permutation matrix
            row, col = linear_sum_assignment(-P_sample.T)

            P_sample = round_to_perm(P_sample)
            print (np.dot(np.arange(P_sample.shape[0]), P_sample), np.dot(np.arange(P_true.shape[0]), P_true))
            print (sp.stats.kendalltau(np.dot(np.arange(P_sample.shape[0]), P_sample), np.dot(np.arange(P_true.shape[0]), P_true)))
            
            ax = fig.add_subplot(4, 4, i*4 + j +1, frameon=True)
            for k in range(K):
                plt.plot(xs[k, 0], xs[k, 1], 'sk', markersize=8)
                plt.plot(mus[k, 0], mus[k, 1], 'ok', markersize=8)

            for k in range(K):
                plt.plot(mus[k, 0], mus[k, 1], 'o',
                         color=colors[k % len(colors)],  markersize=6)
                plt.plot(xs[col[k], 0], xs[col[k], 1], 's',
                         markersize=6, color=colors[k % len(colors)])
                
            # Scale bar
            plt.plot([-5,-5+2*eta], [5,5], '-k', lw=3)

            ax.set_xlim([-5.5, 5.5])
            ax.set_ylim([-5.5, 5.5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Sample {}".format(i*4+j+1))

    plt.tight_layout()
    plt.savefig("permutation_K20_xy.png")

from pybasicbayes.util.text import progprint_xrange
