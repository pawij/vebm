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
from autograd.scipy.special import logsumexp
from autograd import grad
from autograd.misc.optimizers import adam

from birkhoff.primitives import \
    logit, logistic, gaussian_logp, gaussian_entropy, \
    psi_to_birkhoff, log_det_jacobian, birkhoff_to_psi

import scipy as sp

npr.seed(0)

DO_PLOT = False

#PW

import numpy
from sim_funcs import gen_data
n_ppl = 100
n_bms = 10
n_obs = 1
n_components = n_bms+1
model_type = 'GMM'
is_cut = False
sigma_noise = 0.1
fwd_only = False
order = n_bms
scale = 1.
X, lengths, jumps, labels, X0, stages_true, times, seq_true, Q, pi0, _ = gen_data(1, n_ppl, n_bms, n_obs, n_components, model_type=model_type, is_cut=is_cut, n_zscores=None, z_max=None, sigma_noise=sigma_noise, seq=[], fractions=[1], fwd_only=fwd_only, order=order, time_mean=[1/scale])
print ('seq_true', seq_true)
from kde_ebm.mixture_model import fit_all_gmm_models, get_prob_mat
mixtures = fit_all_gmm_models(X0, labels)
prob_mat = get_prob_mat(X, mixtures)
from kde_ebm import plotting, mcmc
fig, ax = plotting.mixture_model_grid(X0, labels, mixtures, np.arange(X0.shape[1]))
#plt.show()
print (mcmc.greedy_ascent_creation(prob_mat)[0][-1])
#res = mcmc.mcmc(X, mixtures, n_iter=100000,
#                greedy_n_iter=1000, greedy_n_init=10)
#fig, ax = plotting.mcmc_uncert_mat(res, score_names=np.arange(X0.shape[1]).astype(str), title='', reverse=True)
#res.sort(reverse=True)    
#ebm_order = res[0]
#print (ebm_order)

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

def perm_to_P(perm):
    K = len(perm)
    P = np.zeros((K, K))
    P[np.arange(K), perm] = 1
    return P

def round_to_perm(P):
    N = P.shape[0]
    assert P.shape == (N, N)
    try:
        row, col = linear_sum_assignment(-P)
    except:
        col = linear_sum_assignment_wrapper(-P)
    P = np.zeros((N, N))
    #    P[row, col] = 1.0
    P[np.arange(N), col] = 1.0
    return P

import torch
def linear_sum_assignment_wrapper(P):
    def hungarian(x):
        if x.ndim == 2:
            x = np.reshape(x, [1, x.shape[0], x.shape[1]])
        sol = np.zeros((x.shape[0], x.shape[1]), dtype=np.int32)
        for i in range(x.shape[0]):
            sol[i, :] = linear_sum_assignment(x[i, :])[1].astype(np.int32)
        return sol
    # trying various things to allow P to be manipulated without removing gradient calculation later
    #    listperms = hungarian(P.detach().cpu().numpy())
    #    listperms = hungarian(P._value)
    #    listperms = torch.from_numpy(listperms)
    #    listperms = hungarian(torch.tensor(P).requires_grad_(False))
    #    listperms = hungarian(P.requires_grad(False))
    #    listperms = listperms.requires_grad_(True)
    #    P_copy = np.array(P, copy=True)
    P_copy = P.astype(np.ndarray, copy=True)
    listperms = hungarian(P_copy._value)
    return listperms

def log_likelihood_ebm(P, t):
    p_yes = np.dot(prob_mat[:, :, 1], P.T)
    p_no = np.dot(prob_mat[:, :, 0], P.T)    
    k = prob_mat.shape[1]+1
    p_perm = []
    for i in range(k):
        p_perm.append(np.prod(p_yes[:, :i], 1)*np.prod(p_no[:, i:k-1], 1))
    ll = np.sum(np.log(np.sum((1./k)*np.array(p_perm), 1)+1e-250))
    return ll

def log_likelihood_ebm_S(S_int):
    p_yes = np.array(prob_mat[:, S_int, 1])
    p_no = np.array(prob_mat[:, S_int, 0])
    k = prob_mat.shape[1]+1
    p_perm = np.zeros((prob_mat.shape[0], k))
    for i in range(k):
        p_perm[:, i] = np.prod(p_yes[:, :i], 1)*np.prod(p_no[:, i:k-1], 1)
    ll = np.sum(np.log(np.sum((1./k)*p_perm, 1)+1e-250))
    return ll

"""
def log_likelihood_ebm_individual(P, i):
    n_samples = 1
    n_features = P.shape[1]
    S_int = np.dot(P, np.arange(n_features)).astype(int)
    
    arange_Np1 = np.arange(0, n_features+1)
    p_perm_k = np.zeros((n_samples, n_features+1))
    p_yes = np.array(prob_mat[i, :, 1]).reshape(n_samples, S_int.shape[0], 1)
    p_no = np.array(prob_mat[i, :, 0]).reshape(n_samples, S_int.shape[0], 1)
    # Leon's clever cumulative probability code
    cp_yes = np.cumprod(p_yes[:, S_int], 1)
    cp_no = np.cumprod(p_no[:, S_int[::-1]], 1)
    for i in arange_Np1:
        if i == 0:
            p_perm_k[:, i] = cp_no[:,n_features-1]
        elif i == n_features:
            p_perm_k[:, i] = cp_yes[:,n_features-1]
        else:
            p_perm_k[:, i] = cp_yes[:,i-1] * cp_no[:,n_features-i-1]
    p_perm_k[p_perm_k==0] = np.finfo(float).eps
    #FIXME: check
    return np.log(np.sum(p_perm_k))
"""
if __name__ == "__main__":
    # Set up a simple matching problem
    is_ebm = 1
    K = 10
    D = 2
    eta = 0.2#0.01
    mus = 2 * npr.randn(K, D)
    num_mcmc_samples = 10
    sigma_min, sigma_max = 1e-3, 5.0

    # Sample a true permutation (in=col, out=row)
    P_true = np.zeros((K, K))
    if is_ebm:
        P_true[np.arange(K), seq_true[0].astype(int)] = 1
    else:
        P_true[np.arange(K), npr.permutation(K)] = 1
        
    # Sample data according to this permutation
    mus_perm = P_true.dot(mus)
    xs = mus_perm + eta * npr.randn(K, D)

    """
    ### greedy routine - for reference to VI
    def greedy_ascent(prob_mat, n_iter=1000, n_init=10):
        n_biomarkers = prob_mat.shape[1]
        starts_dict = dict((x, []) for x in range(n_init))
        mu_start = -2. # magic number
        sigma_start = 5. # magic number
        for start_idx in range(n_init):
            current_mu = mu_start + npr.randn(1, n_biomarkers-1, n_biomarkers-1)[0]
            current_sigma = logistic(sigma_start * npr.randn(1, n_biomarkers-1, n_biomarkers-1)[0])
            Psi = current_mu + current_sigma * npr.randn(1, n_biomarkers-1, n_biomarkers-1)
            P = psi_to_birkhoff(logistic(Psi[0]))
            current_order = np.dot(round_to_perm(P), np.arange(n_biomarkers))
            current_score = log_likelihood_ebm_S(current_order.astype(int))
            starts_dict[start_idx].append([current_order, current_score])
            for iter_n in range(1, n_iter):
                new_mu = (current_mu + npr.randn(1, n_biomarkers-1, n_biomarkers-1))[0]
                new_sigma = logistic(current_sigma * npr.randn(1, n_biomarkers-1, n_biomarkers-1)[0])
                Psi = new_mu + new_sigma * npr.randn(1, n_biomarkers-1, n_biomarkers-1)
                P = psi_to_birkhoff(logistic(Psi[0]))
                new_order = np.dot(round_to_perm(P), np.arange(n_biomarkers))
                new_score = log_likelihood_ebm_S(new_order.astype(int))
                if new_score > current_score:
                    #                    print (current_order, new_order)
                    #                    print (np.min(new_mu), np.max(new_mu), np.mean(new_mu))
                    #                    print (np.min(new_sigma), np.max(new_sigma), np.mean(new_sigma))
                    current_order = new_order
                    current_score = new_score
                    current_mu = new_mu
                    current_sigma = new_sigma
                starts_dict[start_idx].append([current_order, current_score])
        return starts_dict

    n_init = 10
    greedy_dict = greedy_ascent(prob_mat, n_iter=10000, n_init=n_init)
    current_order = greedy_dict[0][-1][0]
    current_like = greedy_dict[0][-1][1]
    fig, ax = plt.subplots()
    for key, value in greedy_dict.items():
        scores = [x[1] for x in value]
        iter_n = np.arange(len(scores))+1
        ax.plot(iter_n, scores, label='Init {}'.format(key+1))
    ax.legend(loc=0)
    fig.suptitle('Greedy Ascent Traces')
    
    for i in range(1, n_init):
        new_order = greedy_dict[i][-1][0]
        new_like = greedy_dict[i][-1][1]
        if new_like > current_like:
            current_order = new_order
            current_like = new_like
    print (current_order, current_like)
    print (log_likelihood_ebm_S(seq_true[0].astype(int)))
    plt.show()
    """
    
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
        # np.dot(P, mus) is similar to S_int = np.dot(P2, np.arange(n_features)).astype(int), i.e., it makes the likelihood dependent on P
        # BUT not the same as have rounded the matrix from doubly stochastic to permutation; does this change the inference?
        # xs, mus, eta, are equivalent to prob_mat, i.e., they are the data generated from the true permutation / sequence
        # BUT not the same as have used different distributions to generate data, as opposed to directly from doubly stochastic matrix parameters
        return np.sum(gaussian_logp(xs, np.dot(P, mus), eta))

    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mu, log_sigma, sigma = unpack_params(params)
        """
        fig, ax = plt.subplots()
        try:
            ax.hist(npr.normal(mu[0,0]._value, sigma[0,0]._value, 100))
        except AttributeError:
            ax.hist(npr.normal(mu[0,0], sigma[0,0], 100))
        plt.show()
        """
        Psi_samples = mu + npr.randn(num_mcmc_samples, K-1, K-1) * sigma
        P_samples = [psi_to_birkhoff(logistic(Psi)) for Psi in Psi_samples]

        # Compute ELBO. Explicitly compute gaussian entropy.
        elbo = 0

        #        elbo = elbo + log_likelihood_ebm(psi_to_birkhoff(logistic((mu + npr.randn(num_mcmc_samples, K-1, K-1) * sigma)[0])), t)
        
        #        for P, Psi in zip(P_samples, Psi_samples):
        for P in P_samples:
            if is_ebm:
                #                print (P.dtype, P)
                elbo = elbo + log_likelihood_ebm(P, t) / num_mcmc_samples
                """
                P_numpy = numpy.asarray(P)
                if P_numpy.dtype == object:
                    temp = []
                    for row in P_numpy:
                        temp.append([x._value for x in row])
                    P = np.array(temp)
                S = np.dot(round_to_perm(P), np.arange(P.shape[0])).astype(int)
                elbo = elbo + log_likelihood_ebm_S(S) / num_mcmc_samples
                """
                #                continue
                #                print (log_likelihood_ebm(P, t) / num_mcmc_samples, log_likelihood_ebm_S(S) / num_mcmc_samples)
            else:
                #                print (P.dtype, P)
                elbo = elbo + log_prob(P, t) / num_mcmc_samples
#            print ('0',elbo)
#            elbo = elbo - log_det_jacobian(P) / num_mcmc_samples
            #            elbo += unconstrained_log_prior(P, 0.01) / num_mcmc_samples
#            print ('1',elbo)
        #        print ('out')
        #        print (elbo)
#        elbo = elbo + gaussian_entropy(log_sigma)
#        print ('2',elbo)
        #        quit()
        # Minimize the negative elbo
        return -elbo# / K

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
        print("mu min: ", mu.min(), "\t mu max: ", mu.max(), "\t mu mean: ", mu.mean())
        print("sigma min: ", sigma.min(), "\t sigma max: ", sigma.max(), "\t sigma mean: ", sigma.mean())

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
    #    init_logit_std = -0.1 * np.ones((K - 1) ** 2)
    init_var_params = np.concatenate([init_mean, init_logit_std])
    variational_params = adam(gradient, init_var_params, step_size=1E-1, num_iters=100, callback=callback)
    # fig.savefig("permutation_K20.png")

    # Plot the elbo
    plt.figure(figsize=(6,4))
    plt.plot(elbos)
    plt.xlim(0, 100)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.tight_layout()
    plt.savefig("permutation_K20_elbo.png")
    
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
            print (np.dot(P_sample, np.arange(P_sample.shape[0])), np.dot(P_true, np.arange(P_true.shape[0])))
            print (sp.stats.kendalltau(np.dot(P_sample, np.arange(P_sample.shape[0])), np.dot(P_true, np.arange(P_true.shape[0]))))
            
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
