import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from vebm import VEBM
from utils import gen_data

try:
    seed = int(sys.argv[1])
except IndexError:
    seed = 42
np.random.seed(seed)

if __name__ == "__main__":
    
    # model parameters
    n_sinkhorn = 20
    temperature = 1E0
    temperature_prior = 1E0
    gumbel_scale = 0
    if gumbel_scale > 0:
        n_mc_samples = 20
    else:
        n_mc_samples = 1
    n_iters = 100
    step_size = 1E-1
        
    # simulate data
    n_ppl = 100
    n_fts = 10
    n_obs = 1
    sigma_noise = 1.0
    sim_file = Path('simdata'+str(seed)+'_n_ppl_'+str(n_ppl)+'_n_fts_'+str(n_fts)+'_sigma_noise_'+str(sigma_noise)+'.csv')    
    print ('Generating simulated data...')
    X, lengths, jumps, labels, X0, stages_true, times, seq_true, Q, pi0, _ = gen_data(n_ppl, n_fts, n_obs, sigma_noise)
    data = {}
    data['X'] = X
    data['labels'] = labels
    data['X0'] = X0
    data['seq_true'] = seq_true
    pickle_file = open(sim_file, 'wb')
    pickle.dump(data, pickle_file)
    pickle_file.close()
    seq_true = seq_true[0]        
    print ('n_ppl {} n_fts {} n_iters {} step_size {} n_sinkhorn {} temperature {} temperature_prior {} gumbel_scale {} n_mc_samples {} sigma_noise {}'.format(n_ppl, n_fts, n_iters, step_size, n_sinkhorn, temperature, temperature_prior, gumbel_scale, n_mc_samples, sigma_noise))
    print ('unique labels', np.unique(labels, return_counts=True))
    
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
                 verbose=True)
    model.train()
    model.plot_sequence(seq_true=seq_true, verbose=True)
    plt.show()
