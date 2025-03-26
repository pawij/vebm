import numpy as np
import scipy as sp
import torch
from torch import logsumexp
#import autograd.numpy as np
from autograd.scipy.special import gammaln
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture as gmm
import pickle

class VEBM(BaseEstimator):

    def __init__(self,
                 X=None,
                 S_prior=None,
                 n_sinkhorn=20,
                 temperature=1E0,
                 temperature_prior=1E0,
                 gumbel_scale=0,
                 n_mc_samples=20,
                 n_iters=100,
                 step_size=1E-1):
        # user-defined variables
        self.X = X
        self.S_prior = S_prior
        self.n_sinkhorn = n_sinkhorn
        self.temperate = temperature
        self.temperate_prior = temperature_prior
        self.gumbel_scale = gumbel_scale
        self.n_mc_samples = n_mc_samples
        self.n_iters = n_iters
        self.step_size = step_size
        # automatically-defined variables
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.device = 'cuda'
            self.dtype = torch.float32
        else:
            self.device = 'cpu'
            self.dtype = torch.float64
        torch.set_default_dtype(self.dtype)
        self.eps = torch.finfo(self.dtype).eps

        def to_var(self, x):
            if self.is_cuda:
                x = x.cuda()
            return x

        def unconstrained_log_prior(self, P, sigmasq):
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

        def perm_to_P(self, perm):
            K = len(perm)
            P = np.zeros((K, K))
            P[np.arange(K), perm] = 1
            return P

        def round_to_perm(self, P):
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

        def vectorised_round_to_perm(self, P):
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
        # Note we omit the uniform prior over k
        #@profile
        def vectorised_log_likelihood_ebm_logspace(self, P):
            k = prob_mat.shape[1]+1
            logp_k = torch.log(torch.tensor(1/k))
            logp_perm_k = torch.zeros((prob_mat.shape[0], k, P.shape[2]))
            p_yes = torch.einsum('ij,jkl->ikl', prob_mat[:, :, 1], P)
            p_yes[p_yes == 0] = eps
            p_no = torch.einsum('ij,jkl->ikl', prob_mat[:, :, 0], torch.flip(P, [1]))
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

        def fit_gmms(self, X, y):
            n_particp, n_biomarkers = X.shape
            mixture_models = []
            for i in range(n_biomarkers):
                bio_y = y[~np.isnan(X[:, i])]
                bio_X = X[~np.isnan(X[:, i]), i]
                mm = gmm(n_components=2, covariance_type='diag', tol=1E-2, n_init=20, means_init=np.array([np.nanmean(bio_X[bio_y==0]), np.nanmean(bio_X[bio_y==1])]).reshape(2,1))#, precisions_init=np.array([1/np.nanstd(bio_X[bio_y==0]), 1/np.nanstd(bio_X[bio_y==1])]).reshape(2,1), weights_init=np.array([0.5,0.5]))
                mm.fit(bio_X.reshape(-1, 1))
                mixture_models.append(mm)
            return mixture_models

        def calc_prob_mat(self, X, y):
            mixtures = self.fit_gmms(X, y)
            prob_mat = np.zeros((X.shape[0], X.shape[1], 2))
            for i in range(X.shape[1]):
                weight = thetas[i].weights_[1]
                con_prob = sp.stats.norm.pdf(X[:,i], loc=mixtures[i].means_[0][0], scale=mixtures[i].covariances_[0][0]) * weight
                cas_prob = sp.stats.norm.pdf(X[:,i], loc=mixtures[i].means_[1][0], scale=mixtures[i].covariances_[1][0]) * (1-weight)
                prob_mat[:, i, 0] = con_prob.flatten()
                prob_mat[:, i, 1] = cas_prob.flatten()
            prob_mat = self.to_var(torch.tensor(prob_mat, dtype=self.dtype))
            return prob_mat
    
        def vectorised_sinkhorn_logspace(self, logP, n_iters=10):
            n = logP.size()[1]
            logP = logP.view(n, n, -1)
            for i in range(n_iters):
                logP = logP - (logsumexp(logP, dim=1, keepdim=True)).view(n, 1, -1)
                logP = logP - (logsumexp(logP, dim=0, keepdim=True)).view(1, n, -1)
            return logP

        def vectorised_sample_gumbel(self, a, temperature, n=1, eps=1E-20):
            return -torch.log(-torch.log(torch.rand((a[0], a[1], n)) + eps) + eps)

        def gumbel_distance(self, log_mu_P, temperature_prior, temperature):
            #FIXME: check
            arr = torch.sum(np.log(temperature_prior) - 0.5772156649 * temperature_prior / temperature -
                            log_mu_P * temperature_prior / temperature -
                            torch.exp(gammaln(1 + temperature_prior / temperature) - log_mu_P * temperature_prior / temperature)
                            - (np.log(temperature) - 1 - 0.5772156649))
            return arr
    
        def variational_objective(self, return_dr=False):
            """Provides a stochastic estimate of the variational lower bound."""
            log_mu_P = self.params[0]
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
    
        def train(self):
            self.params = [self.to_var(torch.zeros((X.shape[1], X.shape[1]), requires_grad=True, device=self.device))]
            optimizer = torch.optim.Adam(self.params, lr=self.step_size, eps=self.eps)
            for i in range(self.n_iters):
                optimizer.zero_grad()
                loss = self.variational_objective()
                #        print (loss)
                loss.backward()
                optimizer.step()

            
