# author: Peter Wijeratne (p.wijeratne@pm.me)
# example VEBM training on simulated data
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from vebm import VEBM
from utils import gen_data

from utils_img import gen_model_mixture_block

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
    sigma_noise = 1E-1 # standard deviation of noise    
    print ('Generating simulated data...')
    # X is observed data for each individual and each observation: shape (n_ppl, n_features, n_obs)
    # X0 is the first observation for each individual only: shape (n_ppl, n_features)
    # labels is the control ("con") or case ("case") labels: shape (n_ppl)
    # S_true is the true simulated sequence, used for post-hoc comparison: shape (n_fts+1)

    gen_model_mixture_block(n_fts, N_groups=2)
    
    X, _, _, labels, X0, stages_true, _, S_true, _, _, _ = gen_data(n_ppl, n_fts, n_obs, sigma_noise)
    print ('n_ppl {} n_fts {} n_iters {} step_size {} n_sinkhorn {} temperature {} temperature_prior {} gumbel_scale {} n_mc_samples {} sigma_noise {}'.format(n_ppl, n_fts, n_iters, step_size, n_sinkhorn, temperature, temperature_prior, gumbel_scale, n_mc_samples, sigma_noise))
    
    # run model
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
    model.plot_sequence(seq_true=S_true[0], verbose=True)
    
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
