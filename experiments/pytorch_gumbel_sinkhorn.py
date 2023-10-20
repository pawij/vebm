import sys
import signal
import warnings
import time
#import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt

import torch
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import gammaln
import scipy as sp

is_cuda = torch.cuda.is_available()

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
    return np.sum(torch.logsumexp(-0.5 * diffs ** 2 / sigmasq, axis=2)) \
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

# Set up the log probability objective
# Assume a uniform prior on P?

def log_likelihood_ebm(P):
    k = prob_mat.shape[1]+1
    p_perm_k = torch.zeros((prob_mat.shape[0], k))
    cp_yes = torch.cumprod(torch.mm(prob_mat[:, :, 1], P.T), 1)
    cp_no = torch.cumprod(torch.mm(prob_mat[:, :, 0], torch.flip(P.T, [1])), 1)
    p_perm_k[:, 0] = cp_no[:, -1]
    p_perm_k[:, 1:-1] = torch.flip(cp_no[:, :-1], [1]) * cp_yes[:, :-1]
    p_perm_k[:, -1] = cp_yes[:, -1]
    p_perm_k /= k
    return torch.sum(torch.log(torch.sum(p_perm_k, 1)+1e-250))
    """
    p_yes = torch.mm(prob_mat[:, :, 1], P.T)
    p_no = torch.mm(prob_mat[:, :, 0], P.T)
    k = prob_mat.shape[1]+1
    p_perm = torch.zeros((k, prob_mat.shape[0]))
    for i in range(k):
        p_perm[i] = torch.prod(p_yes[:, :i], 1)*torch.prod(p_no[:, i:k-1], 1)
    p_perm = p_perm.T
    #    print (p_perm_k)
    #    quit()
    return torch.sum(torch.log(torch.sum((1./k)*p_perm, 1)+1e-250))
    """
if __name__ == "__main__":

    do_mcmc = 0
    do_plot = 0

    n_ppl = 200
    n_bms = 10
    n_obs = 1
    # FIXME: systematically test dependency on these hyperparameters
    num_iters = 100
    step_size = 1E-1
    num_sinkhorn = 10
    temperature = 1.#10.
    temperature_prior = 1.#1.
    gumbel_noise = 0.#0.01
    sigma_start = -2.
    sigmasq_prior = 1.
    if gumbel_noise > 0:
        num_mc_samples = 10
    else:
        num_mc_samples = 1
    data_noise = .1
    #    sigma_min, sigma_max = 1E-8, 1.#1E-3, 5.0
    print ('num_iters, step_size, num_sinkhorn, sigma_start, temperature, temperature_prior, gumbel_noise, num_mc_samples, data_noise', num_iters, step_size, num_sinkhorn, sigma_start, temperature, temperature_prior, gumbel_noise, num_mc_samples, data_noise)
    
    params = [to_var(torch.zeros((n_bms, n_bms), requires_grad=True))]
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
    scale = 1.
    X, lengths, jumps, labels, X0, stages_true, times, seq_true, Q, pi0, _ = gen_data(1, n_ppl, n_bms, n_obs, n_components, model_type=model_type, is_cut=is_cut, n_zscores=n_zscores, z_max=z_max, sigma_noise=data_noise, seq=seq_true, fractions=[1], fwd_only=fwd_only, order=order, time_mean=[1/scale])
    print ('seq_true', seq_true[0].astype(int))

    from kde_ebm.mixture_model import fit_all_gmm_models, get_prob_mat
    from kde_ebm.plotting import plotting
    mixtures = fit_all_gmm_models(X0, labels)
    prob_mat = get_prob_mat(X, mixtures)

    L_yes = torch.zeros((n_ppl, n_bms))
    L_no = torch.zeros((n_ppl, n_bms))
    for i in range(n_bms):
        L_no[:, i], L_yes[:, i] = torch.tensor(mixtures[i].pdf(None, X[:, i]))
    
    if do_plot:    
        fig, ax = plotting.mixture_model_grid(X0, labels, mixtures, np.arange(X0.shape[1]))
    P_true = np.zeros((n_bms, n_bms))
    P_true[np.arange(n_bms), seq_true[0].astype(int)] = 1
        
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

        """
        seen = []
        for x in mcmc_orders:
            if not np.any(np.all(x == seen, axis=1)):
                seen.append(x)
        print (seen)
        quit()
        """
        if do_plot:
            fig, ax = plotting.mcmc_uncert_mat(mcmc_samples, score_names=np.arange(X0.shape[1]).astype(str))#, title='')
            plt.show()
        mcmc_samples.sort(reverse=True)
        ebm_order = mcmc_samples[0]
        kt_mcmc = sp.stats.kendalltau(ebm_order.ordering, seq_true[0])
        print ('S_mcmc, kt_mcmc',ebm_order,kt_mcmc)
        print ('frac_correct', np.sum(ebm_order.ordering==seq_true[0])/n_bms, ' chance ', 1/n_bms)
        quit()
        freq_mcmc = np.zeros(len(seq_true[0]))
        for i in range(len(mcmc_samples)):
            for j in range(len(seq_true[0])):
                if seq_true[0][j] == mcmc_samples[i].ordering[j]:
                    freq_mcmc[j] += 1
        freq_mcmc /= 100000
        #        print (freq_mcmc)
        probright_mcmc, count = 0, 0
        for i in range(len(seq_true[0])):
            if seq_true[0][i] != mcmc_samples[0].ordering[i]:
                probright_mcmc += freq_mcmc[i]
                count += 1
        if count != 0:
            probright_mcmc /= count
        #        print (probright_mcmc)    
        #    plt.show()

    prob_mat = torch.tensor(prob_mat).float()
        
    # Build variational objective.
    def sinkhorn_logspace(logP, n_iters=10):
        n = logP.size()[1]
        logP = logP.view(-1, n, n)
        for i in range(n_iters):
            logP = logP - (torch.logsumexp(logP, dim=2, keepdim=True)).view(-1, n, 1)
            logP = logP - (torch.logsumexp(logP, dim=1, keepdim=True)).view(-1, 1, n)
        return logP
    
    def sample_gumbel(a, temperature, n=1, eps=1E-20):
        return -torch.log(-torch.log(torch.rand((n, a[0], a[1])).float() + eps) + eps)

    def gumbel_distance(log_mu_P, temperature_prior, temperature):
        #FIXME: check
        arr = torch.sum(np.log(temperature_prior) - 0.5772156649 * temperature_prior / temperature -
                        log_mu_P * temperature_prior / temperature -
                        torch.exp(gammaln(1 + temperature_prior / temperature) - log_mu_P * temperature_prior / temperature)
                        - (np.log(temperature) - 1 - 0.5772156649))
        return arr
    
    def variational_objective(params, return_dr=False):
        """Provides a stochastic estimate of the variational lower bound."""
        #FIXME: move out of function scope
        log_mu_P = params[0]
        # calculate ELBO
        distortion, rate = 0., 0.
        for n in range(num_mc_samples):
            log_P = (log_mu_P + sample_gumbel(log_mu_P.shape, temperature)[0] * gumbel_noise) / temperature
            log_P = sinkhorn_logspace(log_P, num_sinkhorn)
            ##Notice how we limit the variance
            # if we also sampled from variance then 'P' could be real valued, which 'log_likelihood_ebm' does not support
            P = torch.exp(log_P)
            # observation
            distortion = distortion + log_likelihood_ebm(P[0]) / num_mc_samples
            # prior?
            #            distortion = distortion + unconstrained_log_prior(P, sigmasq_prior) / num_mc_samples
        """
        log_P = (np.repeat(log_mu_P, num_mc_samples) + sample_gumbel(log_mu_P.shape, temperature, num_mc_samples) * gumbel_noise) / temperature
        def vectorised_gumbel_logspace(log_P, niters):
            for _ in range(niters):
                log_P = log_P - logsumexp(log_P.reshape(log_P.shape[0], log_P.shape[1]*log_P.shape[1]), axis=0, keepdims=True)
                log_P = log_P - logsumexp(log_P.reshape(log_P.shape[0]*log_P.shape[0], log_P.shape[1]), axis=1, keepdims=True)
            return log_P
        def vectorised_log_likelihood_ebm(P, t):
            # check Leon's version
            p_yes = np.dot(prob_mat[:, :, 1], P.T)
            p_no = np.dot(prob_mat[:, :, 0], P.T)
            k = prob_mat.shape[1]+1
            p_perm = []
            for i in range(k):
                p_perm.append(np.prod(p_yes[:, :i], 1)*np.prod(p_no[:, i:k-1], 1))
            p_perm = np.array(p_perm).T
            ll = np.sum(np.log(np.sum((1./k)*p_perm, 1)+1e-250))
            return ll
        log_P = vectorised_gumbel_logspace(log_P, num_sinkhorn)
        P = np.exp(log_P)
        distortion = distortion + vectorised_log_likelihood_ebm(P, t) / num_mc_samples
        """
        # KL divergence
        #FIXME
        rate = gumbel_distance(log_mu_P, temperature_prior, temperature)
        print (distortion, rate)
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
            P_sample = (log_mu_P + sample_gumbel(log_mu_P.shape, temperature)[0] * gumbel_noise) / temperature
            P_sample = sinkhorn_logspace(P_sample, num_sinkhorn)
            ##Notice how we limit the variance
            P_sample = torch.exp(P_sample)
            # Round doubly stochastic matrix P to the nearest permutation matrix
            row, col = torch.linear_sum_assignment(-P_sample)
            num_correct = n_correct(perm_to_P(col.detach().numpy()), P_true)
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
    optimizer = torch.optim.Adam(params, lr=step_size, eps=1e-8)
    for i in range(num_iters):
        # Training phase
        #        model.train()
        optimizer.zero_grad()
        loss = variational_objective(params)
        loss.backward()
        optimizer.step()
    
    t_vi = time.time()-t_start
    print ('after VI', t_vi)
    # fig.savefig("permutation_K20.png")

    # Plot the elbo
    if do_plot:
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
    log_mu_P_post = params[0]

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
        fig, ax = plt.subplots()
        ax.imshow(P_true, interpolation="none", vmin=0, vmax=1)
        ax.set_title("True permutation")

    P_samples, S_samples, kt_samples, num_corrects = [], [], [], []
    freq_vi = np.zeros(len(seq_true[0]))
    #    gumbel_noise = 1.
    print ('Sampling posterior with gumbel noise', gumbel_noise)
    if do_plot:
        fig = plt.figure(figsize=(10, 10), facecolor='white')
    for i in range(4):
        for j in range(4):
            P_sample = (log_mu_P_post + sample_gumbel(log_mu_P_post.shape, temperature)[0] * gumbel_noise) / temperature
            P_sample = sinkhorn_logspace(P_sample, num_sinkhorn)
            ##Notice how we limit the variance
            P_sample = torch.exp(P_sample)
            P_sample = P_sample.detach().numpy()[0]
            P_samples.append(P_sample)
            #            print (np.min(P_sample), np.max(P_sample))

            for k in range(len(seq_true[0])):
                freq_vi[k] += P_sample[k, int(seq_true[0,k])]

            if do_plot:
                ax = fig.add_subplot(4, 4, i*4 + j +1, frameon=True)
                ax.imshow(P_sample, interpolation="none", vmin=0, vmax=1)
                ax.set_title("Sample "+str((i*5)+j))
            
            # Round doubly stochastic matrix P to the nearest permutation matrix
            row, col = sp.optimize.linear_sum_assignment(-P_sample)

            P_sample = round_to_perm(P_sample)
            
            num_correct = n_correct(P_sample, P_true)
            num_corrects.append(num_correct)
            print ('frac_correct', num_correct/n_bms)
            
            S_sample = np.dot(P_sample, np.arange(P_sample.shape[0]))
            S_samples.append(S_sample)
            #            print (np.dot(P_sample, np.arange(P_sample.shape[0])), np.dot(P_true, np.arange(P_true.shape[0])))
            kt_sample = sp.stats.kendalltau(S_sample, seq_true[0].astype(int))
            #            print (kt_sample)
            kt_samples.append(kt_sample)
            
            """
            ax = fig.add_subplot(4, 4, i*4 + j +1, frameon=True)
            for k in range(n_bms):
                plt.plot(X[k, 0], X[k, 1], 'sk', markersize=8)
                plt.plot(mus[k, 0], mus[k, 1], 'ok', markersize=8)

            for k in range(n_bms):
                plt.plot(mus[k, 0], mus[k, 1], 'o',
                         color=colors[k % len(colors)],  markersize=6)
                plt.plot(X[col[k], 0], X[col[k], 1], 's',
                         markersize=6, color=colors[k % len(colors)])
            
            # Scale bar
            plt.plot([-5,-5+2*eta], [5,5], '-k', lw=3)

            ax.set_xlim([-5.5, 5.5])
            ax.set_ylim([-5.5, 5.5])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Sample {}".format(i*4+j+1))
            """
    print ('tot frac_correct',np.mean([x/n_bms for x in num_corrects]), np.std([x/n_bms for x in num_corrects]),' chance ',1/n_bms)
    #    print (np.mean(kt_samples))
    S_unique, counts = np.unique(S_samples, axis=0, return_counts=True)
    print (S_unique, counts)
    S_mode = S_unique[np.argmax(counts)]
    #FIXME
    """
    S_mode = [0]*n_bms
    for i in range(n_bms):
        pos = [0]*n_bms
        for j in range(len(S_samples)):
            pos[np.where(S_samples[j]==i)[0][0]] += 1
        S_mode[np.argmax(pos)] = i
    S_mode = np.array(S_mode)
    """
    confusion_mat = np.zeros((n_bms,n_bms))
    for i in range(n_bms):
        #        confusion_mat[i, :] = np.sum(np.array(S_samples) == S_mode[i], axis=0)
        confusion_mat[i, :] = np.sum(np.array(S_samples) == np.arange(n_bms)[i], axis=0)
    #    S_mode = np.argmax(confusion_mat, axis=0)
    
    kt_vi = sp.stats.kendalltau(S_mode, seq_true[0].astype(int))
    print ('S_true, S_vi, kt_vi', seq_true[0].astype(int), S_mode, kt_vi)
    
    #    if do_plot:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(confusion_mat.T, interpolation='nearest', cmap='Greens')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.sum(P_samples, axis=0), interpolation='nearest', cmap='Greens')
    
    freq_vi /= 16
    #    print (freq_vi)
    probright_vi, count = 0, 0
    for i in range(len(seq_true[0])):
        if seq_true[0][i] != S_mode[i]:
            probright_vi += freq_vi[i]
            count += 1
    if count != 0:
        probright_vi /= count
    #    print (probright_vi)

    if do_mcmc:
        data_out = np.array([t_mcmc, kt_mcmc[0], probright_mcmc, t_vi, kt_vi[0], probright_vi])
        np.savetxt('sim'+str(seed)+'.csv', data_out, delimiter=',')
    else:
        data_out = np.array([np.nan, np.nan, np.nan, t_vi, kt_vi[0], probright_vi])
        np.savetxt('sim'+str(seed)+'.csv', data_out, delimiter=',')
        
    if do_plot:
        plt.show()
    else:
        quit()

    """
    ### ML greedy routine - for reference to VI
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
    
