from scipy.io import loadmat
from mat73 import loadmat as loadmat73
import signal
import warnings
import time
#import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import torch
from torch import logsumexp
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
import scipy as sp
import pandas as pd
#import torchio as tio
import nibabel as nib
import cv2
import sys
from os import listdir
from kde_ebm.mixture_model import fit_all_gmm_models, get_prob_mat, fit_all_kde_models
from kde_ebm.plotting import plotting

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = 'cuda'
    dtype = torch.float32
else:
    device = 'cpu'
    dtype = torch.float64
torch.set_default_dtype(dtype)
eps = torch.finfo(dtype).eps

def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x

try:
    seed = int(sys.argv[1])
except IndexError:
    seed = 42
npr.seed(seed)

def unconstrained_log_prior(P, sigmasq):
    """
    Consider a product (coordinate-wise) of mixtures of
    two gaussians with std sigma_prior and centers at 0 and 1)
    """
    N = P.shape[0]
    assert P.shape == (N, N)
    corners = np.array([0, 1])
    diffs = P[:,:,None] - corners[None, None, :]
    return np.sum(logsumexp(-0.5 * diffs ** 2 / sigmasq, axis=2)) \
        - 0.5 * N**2 * np.log(2 * np.pi) \
        - 0.5 * N**2 * np.log(sigmasq)

def perm_to_P(perm):
    K = len(perm)
    P = np.zeros((K, K))
    P[np.arange(K), perm] = 1
    return P

def round_to_perm(P):
    N = P.shape[0]
    assert P.shape == (N, N)
    try:
        row, col = sp.optimize.linear_sum_assignment(-P)
    except:
        col = linear_sum_assignment_wrapper(-P)
    P = np.zeros((N, N))
    #    P[row, col] = 1.0
    P[np.arange(N), col] = 1.0
    return P

def vectorised_round_to_perm(P):
    N = P.shape[0]
    P_hard = np.empty(P.shape)
    for i in range(P.shape[2]):
        row, col = sp.optimize.linear_sum_assignment(-P[:,:,i])
        #        row, col = solve_dense(-P[:,:,i])
        #        _, row, col = lap.lapjv(-P[:,:,i])
        P_i = np.zeros((N, N))
        P_i[np.arange(N), col] = 1.0
        P_hard[:,:,i] = P_i
    return P_hard

# Set up the log probability objective
# Assume a uniform prior on P?

def log_likelihood_ebm(P):
    k = prob_mat.shape[1]+1
    P_T = torch.permute(P, (1,0))
    p_perm_k = torch.zeros((prob_mat.shape[0], k))
    cp_yes = torch.cumprod(torch.mm(prob_mat[:, :, 1], P_T), 1)
    cp_no = torch.cumprod(torch.mm(prob_mat[:, :, 0], torch.flip(P_T, [1])), 1)
    p_perm_k[:, 0] = cp_no[:, -1]
    p_perm_k[:, 1:-1] = torch.flip(cp_no[:, :-1], [1]) * cp_yes[:, :-1]
    p_perm_k[:, -1] = cp_yes[:, -1]
    return torch.sum(torch.log(torch.sum(p_perm_k, 1)+1e-250))

def vectorised_log_likelihood_ebm(P):
    k = prob_mat.shape[1]+1
    P_T = torch.permute(P, (1,0,2))
    p_perm_k = torch.zeros((prob_mat.shape[0], k, P.shape[2]))
    cp_yes = torch.cumprod(torch.einsum('ij,jkl->ikl', prob_mat[:, :, 1], P_T), axis=1)
    cp_no = torch.cumprod(torch.einsum('ij,jkl->ikl', prob_mat[:, :, 0], torch.flip(P_T, [1])), axis=1)
    p_perm_k[:, 0, :] = cp_no[:, -1, :]
    p_perm_k[:, 1:-1, :] = torch.einsum('ijk,ijk->ijk', torch.flip(cp_no[:, :-1, :], [1]), cp_yes[:, :-1, :])
    p_perm_k[:, -1, :] = cp_yes[:, -1, :]
    return torch.sum(torch.sum(torch.log(torch.sum(p_perm_k, axis=1) + eps), axis=0))

# note we omit the uniform prior over k
def vectorised_log_likelihood_ebm_logspace(P):
    k = prob_mat.shape[1]+1
    P_T = torch.permute(P, (1,0,2))
    logp_k = torch.log(torch.tensor(1/k))
    logp_perm_k = torch.zeros((prob_mat.shape[0], k, P.shape[2]))
    p_yes = torch.einsum('ij,jkl->ikl', prob_mat[:, :, 1], P_T)
    p_yes[p_yes == 0] = eps
    p_no = torch.einsum('ij,jkl->ikl', prob_mat[:, :, 0], torch.flip(P_T, [1]))
    p_no[p_no == 0] = eps
    logp_yes = torch.log(p_yes)
    logp_no = torch.log(p_no)
    logcp_yes = torch.cumsum(logp_yes, axis=1)
    logcp_no = torch.cumsum(logp_no, axis=1)
    logp_perm_k[:, 0, :] = logcp_no[:, -1, :]
    logp_perm_k[:, 1:-1, :] = torch.flip(logcp_no[:, :-1, :], [1]) + logcp_yes[:, :-1, :]
    logp_perm_k[:, -1, :] = logcp_yes[:, -1, :]
    logp_perm = logsumexp(logp_perm_k, axis=1)
    return torch.sum(logp_perm)

if __name__ == '__main__':
    # FIXME: systematically test dependency on these hyperparameters
    num_iters = 100
    step_size = 1E-1
    num_sinkhorn = 10
    temperature = 1.# need to optimise this hyperparameter; higher powers can do better at high noise
    temperature_prior = 1.
    gumbel_scale = .0
    sigmasq_prior = 1.
    if gumbel_scale > 0:
        num_mc_samples = 10
    else:
        num_mc_samples = 1

    #    img_con = loadmat('/home/paww20/data/duke_amd/Farsiu_Ophthalmology_2013_Control_TRT_maps.mat')['Control_TRT_maps']
    #    img_cas = loadmat73('/home/paww20/data/duke_amd/Farsiu_Ophthalmology_2013_AMD_TRT_maps.mat')['AMD_TRT_maps']
    img_con = loadmat('/home/paww20/data/duke_amd/Farsiu_Ophthalmology_2013_Control_RPEDC_maps.mat')['Control_RPEDC_maps']
    img_cas = loadmat73('/home/paww20/data/duke_amd/Farsiu_Ophthalmology_2013_AMD_RPEDC_maps.mat')['AMD_RPEDC_maps']
    print (img_con.shape, img_cas.shape)

    smp_idx = 20
    
    img_con = cv2.resize(img_con, (img_con.shape[1] // smp_idx, img_con.shape[0] // smp_idx), interpolation=cv2.INTER_LINEAR)
    img_cas = cv2.resize(img_cas, (img_cas.shape[1] // smp_idx, img_cas.shape[0] // smp_idx), interpolation=cv2.INTER_LINEAR)
    print (img_con.shape, img_cas.shape)
    
    nan_msk = ~(np.isnan(np.mean(img_con, axis=2)) + np.isnan(np.mean(img_cas, axis=2)))
    #    fig, ax = plt.subplots()
    #    ax.imshow(nan_msk)
    #    plt.show()
    min_con, max_con = np.nanmin(np.mean(img_con, axis=2)), np.nanmax(np.mean(img_con, axis=2))
    min_cas, max_cas = np.nanmin(np.mean(img_cas, axis=2)), np.nanmax(np.mean(img_cas, axis=2))
    fig, ax = plt.subplots()
    ax.imshow(np.mean(img_con, axis=2)*nan_msk, vmin=min_con if min_con < min_cas else min_cas, vmax=min_con if max_con > max_cas else max_cas)
    fig, ax = plt.subplots()
    ax.imshow(np.mean(img_cas, axis=2)*nan_msk, vmin=min_con if min_con < min_cas else min_cas, vmax=min_con if max_con > max_cas else max_cas)   
    
    nx, ny, nz = nan_msk.shape[0], nan_msk.shape[1], 1
    print (nx, ny, nz)

    X, y = [], []
    for i in range(img_con.shape[2]):
        X.append(img_con[:, :, i][nan_msk].ravel())
        y.append(0)
    #
    #    X_cas_mean = np.mean(np.mean(img_cas, axis=2)[nan_msk])
    #    print (X_cas_mean)
    for i in range(img_cas.shape[2]):
        X.append(img_cas[:, :, i][nan_msk].ravel())
        #        if np.mean(X[i]) > X_cas_mean:
        y.append(1)
        #        else:
        #            y.append(2)
    
    """
    # create nan mask over all controls and cases
    nan_msk = ~(np.isnan(np.mean(img_con, axis=2)[::smp_idx,::smp_idx]) + np.isnan(np.mean(img_cas, axis=2)[::smp_idx,::smp_idx]))
    #    fig, ax = plt.subplots()
    #    ax.imshow(nan_msk)
    #    plt.show()
    min_con, max_con = np.nanmin(np.mean(img_con, axis=2)[::smp_idx,::smp_idx]), np.nanmax(np.mean(img_con, axis=2)[::smp_idx,::smp_idx])
    min_cas, max_cas = np.nanmin(np.mean(img_cas, axis=2)[::smp_idx,::smp_idx]), np.nanmax(np.mean(img_cas, axis=2)[::smp_idx,::smp_idx])
    fig, ax = plt.subplots()
    ax.imshow(np.mean(img_con, axis=2)[::smp_idx,::smp_idx]*nan_msk, vmin=min_con if min_con < min_cas else min_cas, vmax=min_con if max_con > max_cas else max_cas)
    fig, ax = plt.subplots()
    ax.imshow(np.mean(img_cas, axis=2)[::smp_idx,::smp_idx]*nan_msk, vmin=min_con if min_con < min_cas else min_cas, vmax=min_con if max_con > max_cas else max_cas)   
    
    nx, ny, nz = nan_msk.shape[0], nan_msk.shape[1], 1
    print (nx, ny, nz)

    X, y = [], []
    for i in range(img_con.shape[2]):
        X.append(img_con[::smp_idx, ::smp_idx, i][nan_msk].ravel())
        y.append(0)
    #
    #    X_cas_mean = np.mean(np.mean(img_cas, axis=2)[::smp_idx,::smp_idx][nan_msk])
    #    print (X_cas_mean)
    for i in range(img_cas.shape[2]):
        X.append(img_cas[::smp_idx, ::smp_idx, i][nan_msk].ravel())
        #        if np.mean(X[i]) > X_cas_mean:
        y.append(1)
        #        else:
        #            y.append(2)
    """
    X, y = np.array(X), np.array(y)
    print (X.shape, y.shape, np.unique(y, return_counts=True), np.min(X[y==0]), np.mean(X[y==0]), np.max(X[y==0]), np.min(X[y==1]), np.mean(X[y==1]), np.max(X[y==1]))#, np.min(X[y==2]), np.mean(X[y==2]), np.max(X[y==2]))
    
    """
    fig, ax = plt.subplots()
    img_con_X = np.zeros(nan_msk.shape)*np.nan
    img_con_X[np.where(nan_msk==True)] = np.mean(X[y==0], axis=0)
    ax.imshow(img_con_X, vmin=min_con if min_con < min_cas else min_cas, vmax=min_con if max_con > max_cas else max_cas)
    fig, ax = plt.subplots()
    img_cas_X = np.zeros(nan_msk.shape)*np.nan
    img_cas_X[np.where(nan_msk==True)] = np.mean(X[y==1], axis=0)
    ax.imshow(img_cas_X, vmin=min_con if min_con < min_cas else min_cas, vmax=min_con if max_con > max_cas else max_cas)
    plt.show()
    """

    del_p, del_m = [], []
    del_p_global = []
    
    del_p = list(np.where(X[0,:] == 0)[0])
    X = np.delete(X, del_p, axis=1)
    print ('Deleted', len(del_p), 'pixels')

    ttest_t = []
    for i in range(X.shape[1]):
        ttest_i = sp.stats.ttest_ind(X[y==0, i], X[y==1, i])
        ttest_t.append(ttest_i[0])
        if np.abs(ttest_t[i]) < 4:
            del_p.append(i)
            del_p_global.append(np.where(nan_msk.ravel()==True)[0][i])
    fig, ax = plt.subplots()
    img_tstat = np.zeros(nan_msk.shape)*np.nan
    img_tstat[np.where(nan_msk==True)] = ttest_t
    ax.imshow(img_tstat)

    fig, ax = plt.subplots()
    X_copy = []
    for i in range(len(X)):
        X_i = X[i]
        X_i[del_p] = np.nan
        X_i_mat = np.zeros(nan_msk.shape)*np.nan
        X_i_mat[np.where(nan_msk==True)] = X_i
        X_copy.append(X_i_mat)
    ax.imshow(np.mean(X_copy, axis=0))

    X = np.delete(X, del_p, axis=1)
    print ('Deleted', len(del_p), 'pixels')

    mixtures = fit_all_gmm_models(X, y, implement_fixed_controls=True)
    #    mixtures = fit_all_kde_models(X, y, implement_fixed_controls=True)
    fig, ax = plotting.mixture_model_grid(X, y, mixtures)
    #    plt.show()
    
    """
    #FIXME: need to preserve original indices in del_m
    tol = 1
    for i,m in enumerate(mixtures):
        if m.theta[1] < tol or m.theta[3] < tol:
            del_m.append(i)
    X = np.delete(X, del_m, axis=1)
    mixtures = np.delete(mixtures, del_m, axis=0)
    print ('Deleted', len(del_m), 'bad mixture models and corresponding pixels')
    """
    # convert prob_mat to torch
    prob_mat = to_var(torch.tensor(get_prob_mat(X, mixtures), dtype=dtype))
    
    # Build variational objective.
    def sinkhorn_logspace(logP, n_iters=10):
        n = logP.size()[1]
        logP = logP.view(-1, n, n)
        for i in range(n_iters):
            logP = logP - (logsumexp(logP, dim=2, keepdim=True)).view(-1, n, 1)
            logP = logP - (logsumexp(logP, dim=1, keepdim=True)).view(-1, 1, n)
        return logP

    def vectorised_sinkhorn_logspace(logP, n_iters=10):
        n = logP.size()[1]
        logP = logP.view(n, n, -1)
        for i in range(n_iters):
            logP = logP - (logsumexp(logP, dim=1, keepdim=True)).view(n, 1, -1)
            logP = logP - (logsumexp(logP, dim=0, keepdim=True)).view(1, n, -1)
        return logP

    def sample_gumbel(a, temperature, n=1, eps=1E-20):
        return -torch.log(-torch.log(torch.rand((n, a[0], a[1])) + eps) + eps)

    def vectorised_sample_gumbel(a, temperature, n=1, eps=1E-20):
        return -torch.log(-torch.log(torch.rand((a[0], a[1], n)) + eps) + eps)

    def gumbel_distance(log_mu_P, temperature_prior, temperature):
        #FIXME: check
        arr = torch.sum(np.log(temperature_prior) - 0.5772156649 * temperature_prior / temperature -
                        log_mu_P * temperature_prior / temperature -
                        torch.exp(gammaln(1 + temperature_prior / temperature) - log_mu_P * temperature_prior / temperature)
                        - (np.log(temperature) - 1 - 0.5772156649))
        return arr
    
    def variational_objective(params, return_dr=False):
        """Provides a stochastic estimate of the variational lower bound."""
        log_mu_P = params[0]
        # vectorise \mu for number of MC samples
        log_mu_P_rep = log_mu_P.unsqueeze(2).repeat(1, 1, num_mc_samples)
        # sample Gumbel noise
        gumbel_noise = to_var(vectorised_sample_gumbel(log_mu_P.shape, temperature, num_mc_samples))
        # add to \mu and scale
        log_P = (log_mu_P_rep + gumbel_noise * gumbel_scale) / temperature
        # move \mu closer to Birkhoff polytope
        log_P = vectorised_sinkhorn_logspace(log_P, num_sinkhorn)
        # note zero variance
        P = torch.exp(log_P)
        # observation likelihood
        distortion = to_var(vectorised_log_likelihood_ebm_logspace(P) / num_mc_samples)
        # KL divergence
        rate = to_var(gumbel_distance(log_mu_P, temperature_prior, temperature))
        # entropy term for \mu?
        if return_dr:
            return -(distortion + rate), distortion, rate
        else:
            return -(distortion + rate)
    
    elbos, sigmas_mean, means_mean, rates, distortions, kt_vi_iter, num_corrects, frac_correct_mean = [], [], [], [], [], [], [], []

    ### Plotting
    """
    if do_plot:
        fig = plt.figure(figsize=(8, 4), facecolor='white')
        ax1 = fig.add_subplot(121, frameon=True)
        ax2 = fig.add_subplot(122, frameon=True)
        plt.ion()
        plt.show(block=False)
    """
    def plot_permutation(ax1, ax2, P):
        ax1.imshow(P_true, interpolation="none", vmin=0, vmax=1)
        ax1.set_title("True $\Pi$")
        ax2.imshow(P, interpolation="none", vmin=0, vmax=1)
        ax2.set_title("Inferred $g(\mu)$")

    def n_correct(P1,P2):
        return P1.shape[0] - np.sum(np.abs(P1-P2))/2.0

    def callback(params, t, g):
        elbo, distortion, rate = variational_objective(params, t, return_dr=True)
        elbos.append(-elbo)
        distortions.append(distortion)
        rates.append(rate)
        print("Iteration {} lower bound {}".format(t, elbos[-1]))
        
        log_mu_P, log_sigmasq_P = unpack_params(params)
        sigma = np.sqrt(np.exp(log_sigmasq_P))
        print("log_mu_P min: ", log_mu_P.min(), "\t log_mu_P max: ", log_mu_P.max(), "\t log_mu_P mean: ", log_mu_P.mean())
        sigmas_mean.append(sigma.mean())
        means_mean.append(log_mu_P.mean())
        
        if ctrlc_pressed[0]:
            sys.exit()

    # Check for quit
    ctrlc_pressed = [False]
    def ctrlc_handler(signal, frame):
        print("Halting due to Ctrl-C")
        ctrlc_pressed[0] = True
    signal.signal(signal.SIGINT, ctrlc_handler)
    
    n_ppl, n_bms = X.shape[0], X.shape[1]    
    params = [to_var(torch.zeros((n_bms, n_bms), requires_grad=True, device=device))]
    
    print("Variational inference for matching...")
    t_start = time.time()
    # FIXME: why does adam work so much better than sgd?
    optimizer = torch.optim.Adam(params, lr=step_size, eps=eps)
    for i in range(num_iters):
        # Training phase
        #        model.train()
        optimizer.zero_grad()
        loss = variational_objective(params)
        print (loss)
        loss.backward()
        optimizer.step()
    
    t_vi = time.time()-t_start
    print ('after VI', t_vi)

    # Sample from the posterior and show samples
    #    gumbel_scale = 1.0
    if gumbel_scale != 0:
        n_samples = 100
    else:
        n_samples = 1
    S_samples = []
    #    log_mu_P_rep = log_mu_P.unsqueeze(2).repeat(1, 1, n_samples)
    for i in range(n_samples):
        log_mu_P = params[0].cpu()
        if is_cuda:
            torch.cuda.empty_cache()
            log_mu_P = log_mu_P.cuda()
        log_mu_P = log_mu_P.unsqueeze(2).repeat(1, 1, 1)        
        # sample Gumbel noise
        gumbel_noise = to_var(vectorised_sample_gumbel(log_mu_P.shape, temperature, 1))
        # add to \mu and scale
        log_P = (log_mu_P + gumbel_noise * gumbel_scale) / temperature
        # move \mu closer to Birkhoff polytope
        log_P = vectorised_sinkhorn_logspace(log_P, num_sinkhorn)
        # note zero variance
        P_samples = torch.exp(log_P)
        P_samples = np.array([x.detach().cpu().numpy() for x in P_samples])
        # round to permutation matrices
        P_hard_samples = vectorised_round_to_perm(P_samples)
        # sequences
        S_samples.append(np.einsum('ijk,j->ik', P_hard_samples, np.arange(n_bms)).T)
    S_unique, counts = np.unique(S_samples, axis=0, return_counts=True)
    S_mode = S_unique[np.argmax(counts)][0].astype(int)
    print ('VI order', S_mode)

    n_pixels = nx*ny*nz
    del_i = list(np.where(nan_msk.ravel()==False)[0]) + del_p_global
    v_events = np.array([i for i in range(n_pixels) if not i in del_i])[S_mode]    
    for i in range(len(S_mode)):
        arr = np.zeros(n_pixels)
        arr[np.where(nan_msk.ravel()==True)[0]] = 1
        arr[v_events[:i+1]] = 2
        # don't know why but FSL transposes and flips the array for these data
        arr = arr.reshape(nx, ny).T
        arr = np.flip(arr, axis=0)
        arr = np.flip(arr, axis=1)
        #
        arr = arr.reshape(nx, ny, nz)
        img = nib.Nifti1Image(arr, affine=np.eye(4))
        if i<10:
            nib.save(img, 'imgs/amd_rpedc_'+str(seed)+'_vi_imgseq_00000'+str(i)+'_nx_'+str(nx)+'_ny_'+str(ny)+'_nz_'+str(nz)+'.nii.gz')
        elif i>=10 and i<100:
            nib.save(img, 'imgs/amd_rpedc_'+str(seed)+'_vi_imgseq_0000'+str(i)+'_nx_'+str(nx)+'_ny_'+str(ny)+'_nz_'+str(nz)+'.nii.gz')
        elif i>=100 and i<1000:
            nib.save(img, 'imgs/amd_rpedc_'+str(seed)+'_vi_imgseq_000'+str(i)+'_nx_'+str(nx)+'_ny_'+str(ny)+'_nz_'+str(nz)+'.nii.gz')
        elif i>=1000 and i<10000:
            nib.save(img, 'imgs/amd_rpedc_'+str(seed)+'_vi_imgseq_00'+str(i)+'_nx_'+str(nx)+'_ny_'+str(ny)+'_nz_'+str(nz)+'.nii.gz')
        elif i>=10000 and i<100000:
            nib.save(img, 'imgs/amd_rpedc_'+str(seed)+'_vi_imgseq_0'+str(i)+'_nx_'+str(nx)+'_ny_'+str(ny)+'_nz_'+str(nz)+'.nii.gz')
        else:
            nib.save(img, 'imgs/amd_rpedc_'+str(seed)+'_vi_imgseq_'+str(i)+'_nx_'+str(nx)+'_ny_'+str(ny)+'_nz_'+str(nz)+'.nii.gz')
    #
    #    quit()
    plt.show()
