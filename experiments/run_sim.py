import sys
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
import numpy as np
#import autograd.numpy as np
#import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
import scipy as sp

from vebm import VEBM

try:
    seed = int(sys.argv[1])
except IndexError:
    seed = 42
np.random.seed(seed)

if __name__ == "__main__":

    n_ppl = 100#816
    n_bms = 10#1344
    n_obs = 1
    # FIXME: systematically test dependency on these hyperparameters
    n_iters = 100
    step_size = 1E-1#1E-6
    n_sinkhorn = 20
    temperature = 1E0# need to optimise this hyperparameter; higher powers can do better at high noise
    temperature_prior = 1E0
    gumbel_scale = 0#1E-9
    sigmasq_prior = 1.
    if gumbel_scale > 0:
        n_mc_samples = 20
    else:
        n_mc_samples = 1
    data_noise = 1.0
    print ('n_ppl {} n_bms {} n_iters {} step_size {} n_sinkhorn {} temperature {} temperature_prior {} gumbel_scale {} n_mc_samples {} data_noise {}'.format(n_ppl, n_bms, n_iters, step_size, n_sinkhorn, temperature, temperature_prior, gumbel_scale, n_mc_samples, data_noise))
    
    #    seq_true = np.array([npr.permutation(n_bms)])
    seq_true = np.array([np.random.permutation(n_bms)])    
    from sim_funcs import gen_data
    model_type = 'GMM'#'Zscore'
    if model_type=='GMM':
        n_zscores = None
        z_max = None
        n_components = n_bms+1
    else:
        n_zscores = 2
        z_max = 3
        n_components = int(n_bms*n_zscores+1)
    is_cut = False
    fwd_only = False
    order = n_bms
    scale = .5
    sim_file = Path('data/simdata'+str(seed)+'_n_ppl_'+str(n_ppl)+'_n_bms_'+str(n_bms)+'_data_noise_'+str(data_noise)+'.csv')
    
    if sim_file.is_file():
        print ('Loading simulated data...')
        pickle_file = open(sim_file, 'rb')
        data = pickle.load(pickle_file)
        X = data['X']
        labels = data['labels']
        X0 = data['X0']
        seq_true = data['seq_true'][0]
        pickle_file.close()
    else:
        print ('Generating simulated data...')
        X, lengths, jumps, labels, X0, stages_true, times, seq_true, Q, pi0, _ = gen_data(1, n_ppl, n_bms, n_obs, n_components, model_type=model_type, is_cut=is_cut, n_zscores=n_zscores, z_max=z_max, sigma_noise=data_noise, seq=seq_true, fractions=[1], fwd_only=fwd_only, order=order, time_mean=[1/scale])
        data = {}
        data['X'] = X
        data['labels'] = labels
        data['X0'] = X0
        data['seq_true'] = seq_true
        pickle_file = open(sim_file, 'wb')
        pickle.dump(data, pickle_file)
        pickle_file.close()
        seq_true = seq_true[0]
    print ('labels', np.unique(labels, return_counts=True))

    elbos, sigmas_mean, means_mean, rates, distortions, kt_vi_iter, n_corrects, frac_correct_mean = [], [], [], [], [], [], [], []
    print("Variational inference for matching...")
    # FIXME: why does adam work so much better than sgd?
    vebm = VEBM(X=X0,
                labels=labels,
                S_prior=None,
                n_sinkhorn=n_sinkhorn,
                temperature=temperature,
                temperature_prior=temperature_prior,
                gumbel_scale=gumbel_scale,
                n_mc_samples=n_mc_samples,
                n_iters=n_iters,
                step_size=step_size)
    vebm.train()
    # Plot the elbo
    # FIXME: these plots need arrays filled in callback function
    if False:#do_plot:
        plt.figure(figsize=(6,4))
        plt.plot(elbos)
        plt.xlim(0, n_iters)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.tight_layout()
        plt.figure(figsize=(6,4))
        plt.plot(sigmas_mean)
        plt.xlim(0, n_iters)
        plt.xlabel("Iteration")
        plt.ylabel("sigmas_mean")
        plt.tight_layout()
        plt.figure(figsize=(6,4))
        plt.plot(means_mean)
        plt.xlim(0, n_iters)
        plt.xlabel("Iteration")
        plt.ylabel("means_mean")
        plt.tight_layout()
        plt.figure(figsize=(6,4))
        plt.plot(distortions)
        plt.xlim(0, n_iters)
        plt.xlabel("Iteration")
        plt.ylabel("distortions")
        plt.tight_layout()
        plt.figure(figsize=(6,4))        
        plt.plot(rates)
        plt.xlim(0, n_iters)
        plt.xlabel("Iteration")
        plt.ylabel("rates")
        plt.tight_layout()
        plt.figure(figsize=(6,4))
        plt.plot(frac_correct_mean)
        plt.xlim(0, n_iters)
        plt.xlabel("Iteration")
        plt.ylabel("frac_correct_mean")
        plt.tight_layout()
    # Sample from the posterior and show samples
    # note that we don't vectorise here for memory reasons
    log_mu_P = vebm.params[0]
    #    print ('log_mu_P',log_mu_P)

    ### point estimate of sequence (zero Gumbel noise)
    # add to \mu and scale
    log_P = (log_mu_P) / temperature
    # move \mu closer to Birkhoff polytope
    log_P = vebm.sinkhorn_logspace(log_P, n_sinkhorn)
    # note zero variance
    P_sample = torch.exp(log_P)
    P_sample = np.array([x.detach().cpu().numpy() for x in P_sample])
    # round to permutation matrices
    P_hard_sample = vebm.round_to_perm(P_sample[0])
    S_point = np.einsum('i,ij->j', np.arange(n_bms), P_hard_sample)

    ### distribution of sequences (non-zero Gumbel noise)
    gumbel_scale = 1E-9
    if gumbel_scale > 0:
        n_samples = 1000
    else:
        n_samples = 1
    S_samples = []
    #    t_start = time.time()
    for i in range(n_samples):
        # sample Gumbel noise
        gumbel_noise = vebm.to_var(vebm.vectorised_sample_gumbel(log_mu_P.shape)[0])
        # add to \mu and scale
        log_P = (log_mu_P + gumbel_noise * gumbel_scale) / temperature
        # move \mu closer to Birkhoff polytope
        log_P = vebm.sinkhorn_logspace(log_P, n_sinkhorn)
        # note zero variance
        P_sample = torch.exp(log_P)
        #        print ('P_sample', P_sample)
        P_sample = np.array([x.detach().cpu().numpy() for x in P_sample])
        #        print ('P_sample[0]', P_sample[0])
        #        fig, ax = plt.subplots()
        #        ax.imshow(P_sample[0], interpolation='nearest', cmap='gray_r')
        # round to permutation matrices
        P_hard_sample = vebm.round_to_perm(P_sample[0])
        #        print ('P_hard_sample', P_hard_sample)
        # sequences
        #        S_samples.append(np.einsum('ij,j->i', P_hard_sample, np.arange(n_bms).T))
        S_samples.append(np.einsum('i,ij->j', np.arange(n_bms), P_hard_sample))        
    S_unique, counts = np.unique(S_samples, axis=0, return_counts=True)
    #    print (S_unique, counts)
    S_mode = S_unique[np.argmax(counts)].astype(int)
    #    t_vi += time.time()-t_start
    #    print ('after VI', t_vi)
    
    confusion_mat = np.zeros((n_bms, n_bms))
    pcorr_vi, ncorr_vi = 0., 0
    for i in range(n_bms):
        confusion_mat[i, :] = np.sum(S_samples == S_point[i], axis=0)
        #        if seq_true[i] != S_point[i]:
        pcorr_vi += np.sum(S_samples == seq_true[i], axis=0)[i]/np.sum(S_samples == seq_true[i])
        ncorr_vi += 1
        #        confusion_mat[i, :] = np.sum(S_samples == np.arange(n_bms)[i], axis=0)
    #    S_mode = np.argmax(confusion_mat, axis=0)

    kt_vi = sp.stats.kendalltau(seq_true.astype(int), S_point)
    print ('S_true, S_vi, kt_vi', seq_true.astype(int), S_point, kt_vi)
    print ('frac_correct', np.sum(S_point==seq_true)/n_bms, ' chance ', 1/n_bms)
    fcorr_vi = np.sum(S_point==seq_true)/n_bms
    if ncorr_vi > 0:
        pcorr_vi /= ncorr_vi
    else:
        pcorr_vi = np.nan
    print ('pcorr_vi', pcorr_vi)

    do_plot = True

    if do_plot:
        """
        fig, ax = plt.subplots()
        ax.imshow(P_true, interpolation='none', vmin=0, vmax=1)
        ax.set_ylabel('Position')
        ax.set_xlabel('Event')
        ax.set_title('True P')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(confusion_mat.T, interpolation='nearest', vmin=0, vmax=1)
        ax.set_ylabel('Position')
        ax.set_xlabel('Event')
        ax.set_title('Inferred P_hard')    

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(np.sum(P_samples, axis=2), interpolation='nearest', vmin=0, vmax=1)
        ax.set_ylabel('Position')
        ax.set_xlabel('Event')
        ax.set_title('Inferred P_soft')
        """
        """
        fig, ax = plt.subplots()
        plt.gca().invert_yaxis()
        ax.scatter(np.mean(S_samples, axis=0), np.arange(n_bms))
        for i, txt in enumerate(np.arange(n_bms).astype(str)):
            ax.annotate(txt, (np.mean(S_samples, axis=0)[i]+0.05, i))
        ax.set_ylabel('Feature', fontsize=20, labelpad=10)
        ax.set_xlabel('Event', fontsize=20)
        ax.set_xticklabels(np.arange(n_bms), fontsize=20)
        ax.set_yticklabels(np.arange(n_bms), fontsize=20)
        if n_bms >= 50 and n_bms < 500:
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_major_ticks()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_major_ticks()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 10 != 0]
        elif n_bms >= 500:
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_major_ticks()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_major_ticks()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 100 != 0]
        plt.subplots_adjust(bottom=0.1, top=0.95)
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(confusion_mat, interpolation='nearest', cmap='gray_r', label='vEBM')
        ax.set_xticks(np.arange(n_bms))
        ax.set_yticks(np.arange(n_bms))
        ax.set_xticklabels(np.arange(n_bms), fontsize=20)
        ax.set_yticklabels(np.arange(n_bms)[S_mode], fontsize=20)
        if n_bms >= 50 and n_bms < 500:
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_major_ticks()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_major_ticks()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 10 != 0]
        elif n_bms >= 500:
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_major_ticks()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_major_ticks()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 100 != 0]
        ax.set_ylabel('Feature', fontsize=20, labelpad=10)
        ax.set_xlabel('Event', fontsize=20)
        for i in range(n_bms):
            if i==0:
                rect = plt.Rectangle((i-.5, np.where(S_mode[i]==seq_true)[0][0]-.5), 1, 1, fill=True, color='black', linewidth=2, label='vEBM')
                ax.add_patch(rect)
                rect = plt.Rectangle((i-.5, np.where(S_mode[i]==seq_true)[0][0]-.5), 1, 1, fill=False, color='red', linewidth=2, label='True')
                ax.add_patch(rect)
            else:
                rect = plt.Rectangle((i-.5, np.where(S_mode[i]==seq_true)[0][0]-.5), 1, 1, fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
        plt.subplots_adjust(bottom=0.1, top=0.95)
        ax.legend(fontsize=20)
    """
    if do_plot:
        sigma_post = np.exp(log_sigmasq_post)
        n_param = log_mu_P_post.flatten().shape[0]
        n_x = np.round(np.sqrt(n_param)).astype(int)
        n_y = np.ceil(np.sqrt(n_param)).astype(int)
        fig, ax = plt.subplots(n_y, n_x, figsize=(12, 12))
        for i in range(n_param):
            ax.flat[i].hist(np.random.normal(log_mu_P_post.flatten()[i], sigma_post.flatten()[i], 1000))
            ax.flat[i].set_xlim(-10,10)
    """        
    if do_plot:
        plt.show()
