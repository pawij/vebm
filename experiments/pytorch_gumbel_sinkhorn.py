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
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
import scipy as sp

is_cuda = False#torch.cuda.is_available()
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

if __name__ == "__main__":

    do_mcmc = 0
    do_plot = 1

    n_ppl = 100#2000
    n_bms = 1000#4096
    n_obs = 1
    # FIXME: systematically test dependency on these hyperparameters
    num_iters = 100
    step_size = 1E-1
    num_sinkhorn = 40
    temperature = 1E1# need to optimise this hyperparameter; higher powers can do better at high noise
    temperature_prior = 1E0
    gumbel_scale = .01
    sigmasq_prior = 1.
    if gumbel_scale > 0:
        num_mc_samples = 100
    else:
        num_mc_samples = 1
    data_noise = 0.1
    print ('n_ppl {} n_bms {} num_iters {} step_size {} num_sinkhorn {} temperature {} temperature_prior {} gumbel_scale {} num_mc_samples {} data_noise {}'.format(n_ppl, n_bms, num_iters, step_size, num_sinkhorn, temperature, temperature_prior, gumbel_scale, num_mc_samples, data_noise))
    
    params = [to_var(torch.zeros((n_bms, n_bms), requires_grad=True, device=device))]
    seq_true = np.array([npr.permutation(n_bms)])
    
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
    sim_file = Path('data/simdata_n_ppl_'+str(n_ppl)+'_n_bms_'+str(n_bms)+'_data_noise_'+str(data_noise)+'.csv')
    
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
    from kde_ebm.mixture_model import fit_all_gmm_models, get_prob_mat
    from kde_ebm.plotting import plotting
    mixtures = fit_all_gmm_models(X0, labels)
    if False:#do_plot:    
        fig, ax = plotting.mixture_model_grid(X0, labels, mixtures, np.arange(X0.shape[1]))
    prob_mat = get_prob_mat(X, mixtures)
    for row in prob_mat:
        if np.any(np.isnan(row)):
            print ('nan in prob_mat!')
            #FIXME: hack
            #            row[np.isnan(row)] = 0.5

    P_true = np.zeros((n_bms, n_bms))
    P_true[np.arange(n_bms), seq_true.astype(int)] = 1
        
    t_start = time.time()    
    if do_mcmc:
        from kde_ebm import mcmc
        #        seq_greedy_ml = mcmc.greedy_ascent_creation(prob_mat)[0][-1]
        #        print (seq_greedy_ml)
        mcmc_samples, greedy_samples = mcmc.mcmc(X, mixtures, n_iter=100000,
                                                 greedy_n_iter=1000, greedy_n_init=10)#, plot=False)
        t_mcmc = time.time()-t_start
        print ('after MCMC',t_mcmc)

        greedy_likes = []
        for i in range(len(greedy_samples)):
            temp = []
            for x in greedy_samples[i]:
                temp.append(x.score)                
            greedy_likes.append(temp)
        greedy_likes = np.mean(greedy_likes, axis=0)
        mcmc_likes = []
        kt_mcmc_iter = []
        for x in mcmc_samples:
            mcmc_likes.append(x.score)
            kt_mcmc_iter.append(sp.stats.kendalltau(x.ordering, seq_true)[0])

        if do_plot:
            fig, ax = plotting.mcmc_uncert_mat(mcmc_samples, score_names=np.arange(X0.shape[1]).astype(str))#, title='')
            plt.show()
        mcmc_samples.sort(reverse=True)
        ebm_order = mcmc_samples[0]
        kt_mcmc = sp.stats.kendalltau(ebm_order.ordering, seq_true)
        print ('S_mcmc, kt_mcmc',ebm_order,kt_mcmc)
        print ('frac_correct', np.sum(ebm_order.ordering==seq_true)/n_bms, ' chance ', 1/n_bms)

    # convert prob_mat to torch
    prob_mat = to_var(torch.tensor(prob_mat, dtype=dtype))
    
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
        #        distortion = vectorised_log_likelihood_ebm(P) / num_mc_samples
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
        print("sigma min: ", sigma.min(), "\t sigma max: ", sigma.max(), "\t sigma mean: ", sigma.mean())

        num_correct_mc = []
        for i in range(num_mc_samples):
            P_sample = (log_mu_P + sample_gumbel(log_mu_P.shape, temperature)[0] * gumbel_scale) / temperature
            P_sample = sinkhorn_logspace(P_sample, num_sinkhorn)
            ##Notice how we limit the variance
            P_sample = torch.exp(P_sample)
            # Round doubly stochastic matrix P to the nearest permutation matrix
            row, col = torch.linear_sum_assignment(-P_sample)
            num_correct = n_correct(perm_to_P(col.detach().cpu().numpy()), P_true)
            num_correct_mc.append(num_correct)
        frac_correct_mean.append(np.mean([x/n_bms for x in num_correct_mc]))
        print ('frac_correct',np.mean([x/n_bms for x in num_correct_mc]), np.std([x/n_bms for x in num_correct_mc]),' chance ',1/n_bms)

        
        #        print (np.dot(perm_to_P(col), np.arange(n_bms)))

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
    # fig.savefig("permutation_K20.png")

    # Plot the elbo
    if False:#do_plot:
        plt.figure(figsize=(6,4))
        plt.plot(elbos)
        plt.xlim(0, num_iters)
        plt.xlabel("Iteration")
        plt.ylabel("ELBO")
        plt.tight_layout()
        plt.figure(figsize=(6,4))
        plt.plot(sigmas_mean)
        plt.xlim(0, num_iters)
        plt.xlabel("Iteration")
        plt.ylabel("sigmas_mean")
        plt.tight_layout()
        plt.figure(figsize=(6,4))
        plt.plot(means_mean)
        plt.xlim(0, num_iters)
        plt.xlabel("Iteration")
        plt.ylabel("means_mean")
        plt.tight_layout()
        plt.figure(figsize=(6,4))
        plt.plot(distortions)
        plt.xlim(0, num_iters)
        plt.xlabel("Iteration")
        plt.ylabel("distortions")
        plt.tight_layout()
        plt.figure(figsize=(6,4))        
        plt.plot(rates)
        plt.xlim(0, num_iters)
        plt.xlabel("Iteration")
        plt.ylabel("rates")
        plt.tight_layout()
        plt.figure(figsize=(6,4))
        plt.plot(frac_correct_mean)
        plt.xlim(0, num_iters)
        plt.xlabel("Iteration")
        plt.ylabel("frac_correct_mean")
        plt.tight_layout()
    """
    fig, ax = plt.subplots()
    #    ax[0].plot(np.linspace(0, t_vi, len(elbos)), elbos)
    ax.plot(np.linspace(0, num_iters, num_iters), elbos, label='VEM')
    all_ebm_iters = list(mcmc_likes)#list(greedy_likes)+list(mcmc_likes)
    ax.plot(np.linspace(0, num_iters, num_iters), all_ebm_iters[:num_iters], label='EBM')
    ax.set_ylabel('Log likelihood', fontsize=20)
    #    ax[0].plot(np.linspace(0, num_iters, num_iters), kt_vi_iter)
    ax.set_xlabel('Iteration', fontsize=20)
    #    ax[1].plot(np.linspace(0, t_mcmc, len(list(greedy_likes)+list(mcmc_likes))), list(greedy_likes)+list(mcmc_likes))
    plt.legend()
    """
    # Sample from the posterior and show samples
    n_samples = 100
#    gumbel_scale = 1.0
    log_mu_P = params[0]
    log_mu_P_rep = log_mu_P.unsqueeze(2).repeat(1, 1, n_samples)
    # sample Gumbel noise
    gumbel_noise = to_var(vectorised_sample_gumbel(log_mu_P.shape, temperature, n_samples))
    # add to \mu and scale
    log_P = (log_mu_P_rep + gumbel_noise * gumbel_scale) / temperature
    # move \mu closer to Birkhoff polytope
    log_P = vectorised_sinkhorn_logspace(log_P, num_sinkhorn)
    # note zero variance
    P_samples = torch.exp(log_P)
    P_samples = np.array([x.detach().cpu().numpy() for x in P_samples])
    # round to permutation matrices
    P_hard_samples = vectorised_round_to_perm(P_samples)
    # sequences
    S_samples = np.einsum('ijk,j->ik', P_hard_samples, np.arange(n_bms)).T
    S_unique, counts = np.unique(S_samples, axis=0, return_counts=True)
    #    print (S_unique, counts)
    S_mode = S_unique[np.argmax(counts)].astype(int)
    
    confusion_mat = np.zeros((n_bms, n_bms))
    for i in range(n_bms):
        confusion_mat[i, :] = np.sum(S_samples == S_mode[i], axis=0)
        #        confusion_mat[i, :] = np.sum(S_samples == np.arange(n_bms)[i], axis=0)
    #    S_mode = np.argmax(confusion_mat, axis=0)

    kt_vi = sp.stats.kendalltau(S_mode, seq_true.astype(int))
    print ('S_true, S_vi, kt_vi', seq_true.astype(int), S_mode, kt_vi)
    print ('frac_correct', np.sum(S_mode==seq_true)/n_bms, ' chance ', 1/n_bms)

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
    if do_mcmc:
        data_out = np.array([t_mcmc, kt_mcmc[0], t_vi, kt_vi[0]])
        np.savetxt('sim'+str(seed)+'.csv', data_out, delimiter=',')
    else:
        data_out = np.array([np.nan, np.nan, t_vi, kt_vi[0]])
        np.savetxt('sim'+str(seed)+'.csv', data_out, delimiter=',')
        
    if do_plot:
        plt.show()
    else:
        quit()
