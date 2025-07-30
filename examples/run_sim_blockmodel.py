# author: Peter Wijeratne (p.wijeratne@pm.me)
# example VEBM training on simulated data
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from vebm import VEBM
from utils_img import gen_data

try:
    seed = int(sys.argv[1])
except IndexError:
    seed = 42
np.random.seed(seed)

if __name__ == "__main__":
    
    # model parameters
    n_sinkhorn = 20 # number of Sinkhorn-Knopp iterations
    temperature = 1E0 # temperature hyperparameter
    temperature_prior = 1E0 # temperature prior hyperparameter
    gumbel_scale = 0 # Gumbel noise hyperparameter
    if gumbel_scale > 0:
        n_mc_samples = 100 # number of Monte Carlo samples
    else:
        n_mc_samples = 1
    n_iters = 100 # number of ADAM iterations
    step_size = 1E-1 # step size for ADAM
        
    # simulate data
    n_ppl = 100 # number of individuals
    n_fts = 4 # number of features
    n_obs = 1 # number of observations per individual
    sigma_noise = 1E0 # standard deviation of noise    
    print ('Generating simulated data...')
    # X is observed data for each individual and each observation: shape (n_ppl, n_features, n_obs)
    # X0 is the first observation for each individual only: shape (n_ppl, n_features)
    # labels is the control ("con") or case ("case") labels: shape (n_ppl)
    # S_true is the true simulated sequence, used for post-hoc comparison: shape (n_fts+1)

    X, _, _, labels, X0, stages_true, _, S_true, _, _, _ = gen_data(n_ppl, n_fts, n_obs, sigma_noise, n_groups=2, n_subtypes=1, model_type='block')
    print ('n_ppl {} n_fts {} n_iters {} step_size {} n_sinkhorn {} temperature {} temperature_prior {} gumbel_scale {} n_mc_samples {} sigma_noise {}'.format(n_ppl, n_fts, n_iters, step_size, n_sinkhorn, temperature, temperature_prior, gumbel_scale, n_mc_samples, sigma_noise))
    """
    print (X.shape, X0.shape, labels.shape, stages_true.shape, S_true.shape)
    print (np.unique(labels, return_counts=True))
    n_x = np.round(np.sqrt(n_fts)).astype(int)
    n_y = np.ceil(np.sqrt(n_fts)).astype(int)
    fig, ax = plt.subplots(n_y, n_x, figsize=(12, 12))
    for i in range(n_fts):
        bio_X = X0[:, i]
        bio_y = labels[~np.isnan(bio_X)]
        bio_X = bio_X[~np.isnan(bio_X)]
        hist_dat = [bio_X[bio_y == 0],
                    bio_X[bio_y == 1]]
        n_unique_values_bio_X = len(np.unique(bio_X))
        leg1 = ax.flat[i].hist(hist_dat,
                               label=['Con','Cas'],
                               density=True,
                               alpha=0.7,
                               stacked=True)
        ax.flat[i].axes.get_yaxis().set_visible(False)
    plt.show()
    """
    # run model
    """
    #TODO: add group variable to likelihood code    
    ####
    #### if 1 event / group, should return same likelihood as original non-block model
    #### i.e., if B_mat == Id && G_mat == Id
    ####
    #Q: calculate likelihood in 1 or 2 operations?
    #1: i) operate on data likelihood to group and order terms according to G and S (i.e., sparsity inducing transform on data likelihood matrices)
    ### no - because you would need every permutation within each group (I think)
    #2: i) operate on data likelihood to group terms according to G (i.e, change shape of data likelihood from (,N_biomarkers) to (,N_groups))
    ### e.g., [1, 2, 3] (shape: (1,3)) --> [[1, 2], [3]] (shape (inhomogeneous): (1,2,N_per_group))    
    #2: ii) operate on i) to order terms according to S
    #### inference steps
    # fix B_mat throughout inference (fit it to some dataset first)
    # generative model is dependent on S_mat, G_mat only (not B_mat)
    # B_mat appears in ELBO via KL term on A_mat
    ####
    p_yes = np.array([[1, 2, 3]])
    # hard assignment
    # FIXME: inhomogeneous shape
    p_yes_block = [p_yes[0][G_mat.T[j]==1] for j in range(N_groups)]
    print (p_yes_block)
    # soft assignment
    p_yes_block = np.array([np.multiply(p_yes[0],G_mat.T[j]) for j in range(N_groups)]).T
    print (p_yes_block)
    print (p_yes_block.shape, S_mat.shape)
    # order likelihoods
    p_yes_block = np.einsum('ij,jk->ik', p_yes_block, S_mat)
    print (p_yes_block.T)
    import torch
    print (torch.tensor(p_yes_block, dtype=torch.float64))
    """
    print("Variational inference for matching...")
    model = VEBM(X=X0,
                 labels=labels,
                 n_sinkhorn=n_sinkhorn,
                 temperature=temperature,
                 temperature_prior=temperature_prior,
                 gumbel_scale=gumbel_scale,
                 n_mc_samples=n_mc_samples,
                 n_iters=n_iters,
                 step_size=step_size,
                 use_em=True,
                 verbose=True)
    model.train()
    model.plot_gmms()
    #    model.plot_sequence(seq_true=S_true, verbose=True)
    model.plot_sequence(verbose=True, gumbel_scale=1E-1)
    plt.show()
    
    stages, stage_likes = model.predict_stage(X0, hard_perm=True)
    fig, ax = plt.subplots()
    ax.hist([stages[labels==0],stages[labels==1],stages[labels==2]], bins=X.shape[1]+1)

    fig, ax = plt.subplots()
    scale = [10.]*len(stages_true)
    for i in range(len(stages_true)):
        x0 = stages_true[i]
        x1 = stages[i]
        for j in range(len(stages_true)):
            if x0 == stages_true[j] and x1 == stages[j]:
                scale[i] += 20.
    ax.scatter(stages_true, stages, s=scale)
    ax.set_xlabel('Stage (true)')
    ax.set_ylabel('Stage (reco)')
    ax.grid()
    print (np.sum(stages==stages_true)/stages_true.shape[0])    

    plt.show()
