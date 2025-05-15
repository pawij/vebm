# author: Peter Wijeratne (p.wijeratne@pm.me)
# VEBM class
import numpy as np
import scipy as sp
from scipy.special import gammaln
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture as gmm
import torch
from torch import logsumexp
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

class VEBM(BaseEstimator):

    def __init__(self,
                 X=None,
                 labels=None,
                 S_prior=None,
                 n_sinkhorn=20,
                 temperature=1E0,
                 temperature_prior=1E0,
                 gumbel_scale=0,
                 n_mc_samples=20,
                 n_iters=100,
                 step_size=1E-1,
                 sigmasq_prior=1.0,
                 use_em=True,
                 verbose=False):
        
        # user-defined variables
        self.X = X
        self.labels = labels
        self.S_prior = S_prior
        self.n_sinkhorn = n_sinkhorn
        self.temperature = temperature
        self.temperature_prior = temperature_prior
        self.gumbel_scale = gumbel_scale
        self.n_mc_samples = n_mc_samples
        self.n_iters = n_iters
        self.step_size = step_size
        self.use_em = use_em
        #FIXME: not currently used in ELBO
        #        self.sigmasq_prior = sigmasq_prior
        self.verbose = verbose
        
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
        #FIXME: assume a uniform prior on P?
        self.params = [self.to_var(torch.zeros((self.X.shape[1], self.X.shape[1]), requires_grad=True, device=self.device))]
        
    def to_var(self, x):
        if self.is_cuda:
            x = x.cuda()
        return x

    def vectorised_log_likelihood_ebm_logspace(self, P):
        k = self.prob_mat.shape[1]+1
        # note we omit the uniform prior over k
        #        logp_k = torch.log(torch.tensor(1/k))
        logp_perm_k = torch.zeros((self.prob_mat.shape[0], k, P.shape[2]))
        p_yes = torch.einsum('ij,jkl->ikl', self.prob_mat[:, :, 1], P)
        p_yes[p_yes == 0] = self.eps
        p_no = torch.einsum('ij,jkl->ikl', self.prob_mat[:, :, 0], torch.flip(P, [1]))
        p_no[p_no == 0] = self.eps
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
        thetas = []
        if self.use_em:
            for i in range(X.shape[1]):
                y_i = y[~np.isnan(X[:, i])]
                X_i = X[~np.isnan(X[:, i]), i]
                mm = gmm(n_components=2, covariance_type='diag', tol=1E-3, n_init=100,
                         means_init=np.array([np.nanmean(X_i[y_i==0]), np.nanmean(X_i[y_i==1])]).reshape(2,1),
                         precisions_init=np.array([1/np.nanstd(X_i[y_i==0])**2, 1/np.nanstd(X_i[y_i==1])**2]).reshape(2,1),
                         weights_init=np.array([0.5,0.5]))
                mm.fit(X_i.reshape(-1, 1))
                thetas.append([mm.means_[0][0], np.sqrt(mm.covariances_[0][0]),
                               mm.means_[1][0], np.sqrt(mm.covariances_[1][0]),
                               mm.weights_[0]])
        else:
            def gmm_like(theta, X, y):
                pdf_0 = sp.stats.norm.pdf(X[y==0], loc=theta[0], scale=theta[1]) * theta[4]
                pdf_1 = sp.stats.norm.pdf(X[y==1], loc=theta[2], scale=theta[3]) * (1-theta[4])
                pdf_0[np.isnan(pdf_0)] = .5
                pdf_1[np.isnan(pdf_1)] = .5
                like = np.concatenate((pdf_0, pdf_1))
                like[like == 0] = np.finfo(float).eps
                if np.sum(np.isnan(like))>0 or np.sum(np.isinf(like))>0:
                    quit()
                return -1*np.sum(np.log(like))
            for i in range(X.shape[1]):
                y_i = y[~np.isnan(X[:, i])]
                X_i = X[~np.isnan(X[:, i]), i]
                theta_0 = [np.nanmean(X_i[y_i==0]), np.nanstd(X_i[y_i==0]), np.nanmean(X_i[y_i==1]), np.nanstd(X_i[y_i==1]), 0.5]
                fit = sp.optimize.minimize(gmm_like,
                                           theta_0,
                                           args=(X_i, y_i),
                                           method='SLSQP')
                thetas.append(fit.x)
        self.thetas = thetas

    def calc_prob_mat(self, X):
        prob_mat = np.zeros((X.shape[0], X.shape[1], 2))
        for i in range(X.shape[1]):
            pdf_0 = sp.stats.norm.pdf(X[:,i], loc=self.thetas[i][0], scale=self.thetas[i][1])
            pdf_1 = sp.stats.norm.pdf(X[:,i], loc=self.thetas[i][2], scale=self.thetas[i][3])
            pdf_0[np.isnan(pdf_0)] = 0.5
            pdf_1[np.isnan(pdf_1)] = 0.5
            prob_mat[:, i, 0] = pdf_0.flatten()
            prob_mat[:, i, 1] = pdf_1.flatten()
        self.prob_mat = self.to_var(torch.tensor(prob_mat, dtype=self.dtype))

    def sinkhorn_logspace(self, logP, n_iters=10):
        n = logP.size()[1]
        logP = logP.view(-1, n, n)
        for i in range(n_iters):
            logP = logP - (logsumexp(logP, dim=2, keepdim=True)).view(-1, n, 1)
            logP = logP - (logsumexp(logP, dim=1, keepdim=True)).view(-1, 1, n)
        return logP
    
    def vectorised_sinkhorn_logspace(self, logP, n_iters=10):
        n = logP.size()[1]
        logP = logP.view(n, n, -1)
        for i in range(n_iters):
            logP = logP - (logsumexp(logP, dim=1, keepdim=True)).view(n, 1, -1)
            logP = logP - (logsumexp(logP, dim=0, keepdim=True)).view(1, n, -1)
        return logP

    def sample_gumbel(self, P, n=1):
        return -torch.log(-torch.log(torch.rand((n, P[0], P[1])) + self.eps) + self.eps)

    def vectorised_sample_gumbel(self, P, n=1):
        return -torch.log(-torch.log(torch.rand((P[0], P[1], n)) + self.eps) + self.eps)

    def gumbel_distance(self, log_mu_P):
        # from https://arxiv.org/abs/1802.08665 Supplementary Section B.3
        # note the seemingly magic number comes from the Gumbel distribution expectation (which is equal to the Euler-Mascheroni constant: https://en.wikipedia.org/wiki/Euler%27s_constant)
        arr = torch.sum(np.log(self.temperature_prior) - 0.5772156649 * self.temperature_prior / self.temperature -
                        log_mu_P * self.temperature_prior / self.temperature -
                        torch.exp(gammaln(1 + self.temperature_prior / self.temperature) - log_mu_P * self.temperature_prior / self.temperature)
                        - (np.log(self.temperature) - 1 - 0.5772156649))
        return arr
    """
    #FIXME: not currently used in ELBO
    def unconstrained_log_prior(self, P):
        N = P.shape[0]
        assert P.shape == (N, N)
        corners = np.array([0, 1])
        diffs = P[:,:,None] - corners[None, None, :]
        return np.sum(logsumexp(-0.5 * diffs ** 2 / self.sigmasq_prior, axis=2)) \
            - 0.5 * N**2 * np.log(2 * np.pi) \
            - 0.5 * N**2 * np.log(self.sigmasq_prior)
    """
    def variational_objective(self, return_dr=False):
        log_mu_P = self.params[0]
        # vectorise \mu for number of MC samples
        log_mu_P_rep = log_mu_P.unsqueeze(2).repeat(1, 1, self.n_mc_samples)
        # sample Gumbel noise
        gumbel_noise = self.to_var(self.vectorised_sample_gumbel(log_mu_P.shape, self.n_mc_samples))
        # add to \mu and scale
        log_P = (log_mu_P_rep + gumbel_noise * self.gumbel_scale) / self.temperature
        # move \mu closer to Birkhoff polytope
        log_P = self.vectorised_sinkhorn_logspace(log_P, self.n_sinkhorn)
        # note zero variance
        P = torch.exp(log_P)
        # observation likelihood
        distortion = self.to_var(self.vectorised_log_likelihood_ebm_logspace(P) / self.n_mc_samples)
        # KL divergence
        rate = self.to_var(self.gumbel_distance(log_mu_P))
        # FIXME: entropy term for \mu?
        if return_dr:
            return -(distortion + rate), distortion, rate
        else:
            return -(distortion + rate)
            
    def train(self):
        self.fit_gmms(self.X, self.labels)
        self.calc_prob_mat(self.X)
        optimizer = torch.optim.Adam(self.params, lr=self.step_size, eps=self.eps)
        for i in range(self.n_iters):
            optimizer.zero_grad()
            loss = self.variational_objective()
            if self.verbose:
                print (loss)
            loss.backward()
            optimizer.step()

    def predict_stage(self, X, hard_perm=True):
        self.calc_prob_mat(X)
        log_mu_P = self.params[0]
        # point estimate of sequence (zero Gumbel noise)
        # add to \mu and scale
        log_P = (log_mu_P) / self.temperature
        # move \mu closer to Birkhoff polytope
        log_P = self.sinkhorn_logspace(log_P, self.n_sinkhorn)
        # note zero variance
        P_soft = torch.exp(log_P)
        k = self.prob_mat.shape[1]+1
        if hard_perm:
            # round to permutation matrices
            P_soft = P_soft.detach().cpu().numpy()
            P_hard = self.round_to_perm(P_soft[0])
            S_hard = np.einsum('i,ij->j', np.arange(X.shape[1]), P_hard)
            p_yes = np.array(self.prob_mat[:, S_hard, 1])
            p_yes[p_yes == 0] = self.eps
            p_no = np.array(self.prob_mat[:, S_hard, 0])
            p_no[p_no == 0] = self.eps
            logp_yes = np.log(p_yes)
            logp_no = np.log(p_no)
            logcp_yes = np.cumsum(logp_yes, axis=1)
            logcp_no = np.cumsum(logp_no, axis=1)
            logp_perm_k = np.zeros((self.prob_mat.shape[0], k))
            logp_perm_k[:, 0] = logcp_no[:, -1]
            logp_perm_k[:, 1:-1] = np.flip(logcp_no[:, :-1], [1]) + logcp_yes[:, :-1]
            logp_perm_k[:, -1] = logcp_yes[:, -1]
            #            p_perm_k = np.exp(logp_perm_k)/np.sum(np.exp(logp_perm_k), axis=1).reshape(logp_perm_k.shape[0], 1)
        else:
            #FIXME
            print ('Not implemented')
            quit()
            p_yes = torch.einsum('ij,jkl->ikl', self.prob_mat[:, :, 1], P_soft)
            p_yes[p_yes == 0] = self.eps
            p_no = torch.einsum('ij,jkl->ikl', self.prob_mat[:, :, 0], torch.flip(P_soft, [1]))
            p_no[p_no == 0] = self.eps
            p_yes = p_yes.detach().numpy()
            p_no = p_no.detach().numpy()
            logp_yes = np.log(p_yes)
            logp_no = np.log(p_no)
            logcp_yes = np.cumsum(logp_yes, axis=1)
            logcp_no = np.cumsum(logp_no, axis=1)
            logp_perm_k = np.zeros((self.prob_mat.shape[0], k, P_soft.shape[2]))
            logp_perm_k[:, 0, :] = logcp_no[:, -1, :]
            logp_perm_k[:, 1:-1, :] = np.flip(logcp_no[:, :-1, :], [1]) + logcp_yes[:, :-1, :]
            logp_perm_k[:, -1, :] = logcp_yes[:, -1, :]
            logp_perm = logsumexp(logp_perm_k, axis=1)
            # print (logp_perm_k.shape) (n_ppl, n_events, P_soft.shape[0]*P_soft.shape[1])
        #
        stages = np.argmax(logp_perm_k, axis=1)
        """
        stages = np.zeros(p_perm_k.shape[0])
        for i in range(p_perm_k.shape[0]):
            stages[i] = np.mean(p_perm_k[i] * np.arange(1,k+1)) * (k-1) - 1
        """
        return stages, logp_perm_k
        """
        # note we omit the uniform prior over k
        #        logp_k = torch.log(torch.tensor(1/k))
        logp_perm_k = torch.zeros((self.prob_mat.shape[0], k, P_soft.shape[2]))
        p_yes = torch.einsum('ij,jkl->ikl', self.prob_mat[:, :, 1], P_soft)
        p_yes[p_yes == 0] = self.eps
        p_no = torch.einsum('ij,jkl->ikl', self.prob_mat[:, :, 0], torch.flip(P_soft, [1]))
        p_no[p_no == 0] = self.eps
        logp_yes = np.log(p_yes.detach().cpu().numpy())
        logp_no = np.log(p_no.detach().cpu().numpy())

        logp_yes_vec = np.zeros((logp_yes.shape[0], logp_yes.shape[1]))
        logp_no_vec = np.zeros((logp_no.shape[0], logp_no.shape[1]))
        for i in range(logp_yes_vec.shape[0]):
            val_yes_i, vec_yes_i = np.linalg.eig(logp_yes[i])
            logp_yes_vec[i] = val_yes_i
            val_no_i, vec_no_i = np.linalg.eig(logp_no[i])
            logp_no_vec[i] = val_no_i
            
        stage_likes = np.zeros((X.shape[0], k))
        for i in range(k):
            stage_likes[:, i] = np.nanprod(logp_yes_vec[:, :i], 1)*np.nanprod(logp_no_vec[:, i:X.shape[1]], 1)
        stages = np.argmax(stage_likes, axis=1)
        """
        return stages, stage_likes

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

    def plot_sequence(self, seq_true=[], gumbel_scale=None, verbose=False):
        n_feat = self.X.shape[1]
        log_mu_P = self.params[0]
        # point estimate of sequence (zero Gumbel noise)
        # add to \mu and scale
        log_P = (log_mu_P) / self.temperature
        # move \mu closer to Birkhoff polytope
        log_P = self.sinkhorn_logspace(log_P, self.n_sinkhorn)
        # note zero variance
        P_sample = torch.exp(log_P)
        P_sample = np.array([x.detach().cpu().numpy() for x in P_sample])
        # round to permutation matrices
        P_hard_sample = self.round_to_perm(P_sample[0])
        S_point = np.einsum('i,ij->j', np.arange(n_feat), P_hard_sample)

        # distribution of sequences (non-zero Gumbel noise)
        if not gumbel_scale:
            gumbel_scale = self.gumbel_scale
        if gumbel_scale > 0:
            n_samples = 1000
        else:
            n_samples = 1
        S_samples = []
        # note that we don't vectorise here for memory reasons; but we could
        for i in range(n_samples):
            # sample Gumbel noise
            gumbel_noise = self.to_var(self.sample_gumbel(log_mu_P.shape)[0])
            # add to \mu and scale
            log_P = (log_mu_P + gumbel_noise * gumbel_scale) / self.temperature
            # move \mu closer to Birkhoff polytope
            log_P = self.sinkhorn_logspace(log_P, self.n_sinkhorn)
            # note zero variance
            P_sample = torch.exp(log_P)
            P_sample = np.array([x.detach().cpu().numpy() for x in P_sample])
            # round to permutation matrices
            P_hard_sample = self.round_to_perm(P_sample[0])
            # sequences
            S_samples.append(np.einsum('i,ij->j', np.arange(n_feat), P_hard_sample))        
        S_unique, counts = np.unique(S_samples, axis=0, return_counts=True)
        S_mode = S_unique[np.argmax(counts)].astype(int)
    
        confusion_mat = np.zeros((n_feat, n_feat))
        pcorr_vi, ncorr_vi = 0., 0
        for i in range(n_feat):
            confusion_mat[i, :] = np.sum(S_samples == S_point[i], axis=0)
            if len(seq_true)>0:
                pcorr_vi += np.sum(S_samples == seq_true[i], axis=0)[i]/np.sum(S_samples == seq_true[i])
                ncorr_vi += 1
        if len(seq_true)>0:
            kt_vi = sp.stats.kendalltau(seq_true.astype(int), S_point)
            fcorr_vi = np.sum(S_point==seq_true)/n_feat
            if ncorr_vi > 0:
                pcorr_vi /= ncorr_vi
            else:
                pcorr_vi = np.nan
            if verbose and len(seq_true)>0:
                print ('S_true, S_vi, kt_vi', seq_true.astype(int), S_point, kt_vi)
                print ('frac_correct', np.sum(S_point==seq_true)/n_feat, ' chance ', 1/n_feat)
                print ('pcorr_vi', pcorr_vi)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(confusion_mat, interpolation='nearest', cmap='gray_r', label='Reco')
        ax.set_xticks(np.arange(n_feat))
        ax.set_yticks(np.arange(n_feat))
        ax.set_xticklabels(np.arange(n_feat), fontsize=20)
        ax.set_yticklabels(np.arange(n_feat)[S_mode], fontsize=20)
        if n_feat >= 50 and n_feat < 500:
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_major_ticks()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_major_ticks()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 10 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 10 != 0]
        elif n_feat >= 500:
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_major_ticks()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_major_ticks()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 100 != 0]
            [l.set_visible(False) for (i,l) in enumerate(ax.yaxis.get_ticklabels()) if i % 100 != 0]
        ax.set_ylabel('Feature', fontsize=20, labelpad=10)
        ax.set_xlabel('Event', fontsize=20)
        if len(seq_true)>0:
            for i in range(n_feat):
                if i==0:
                    rect = plt.Rectangle((i-.5, np.where(S_mode[i]==seq_true)[0][0]-.5), 1, 1, fill=True, color='black', linewidth=2, label='Reco')
                    ax.add_patch(rect)
                    rect = plt.Rectangle((i-.5, np.where(S_mode[i]==seq_true)[0][0]-.5), 1, 1, fill=False, color='red', linewidth=2, label='True')
                    ax.add_patch(rect)
                else:
                    rect = plt.Rectangle((i-.5, np.where(S_mode[i]==seq_true)[0][0]-.5), 1, 1, fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
            ax.legend(fontsize=20)
        plt.subplots_adjust(bottom=0.15, top=0.95)

    def plot_gmms(self, score_names=None, class_names=None):
        n_particp, n_biomarkers = self.X.shape
        if score_names is None:
            score_names = ['BM{}'.format(x+1) for x in range(n_biomarkers)]
        if class_names is None:
            class_names = ['Control', 'Case']
        n_x = np.round(np.sqrt(n_biomarkers)).astype(int)
        n_y = np.ceil(np.sqrt(n_biomarkers)).astype(int)
        fig, ax = plt.subplots(n_y, n_x, figsize=(12, 12))
        for i in range(n_biomarkers):
            bio_X = self.X[:, i]
            bio_y = self.labels[~np.isnan(bio_X)]
            bio_X = bio_X[~np.isnan(bio_X)]
            hist_dat = [bio_X[bio_y == 0],
                        bio_X[bio_y == 1]]
            n_unique_values_bio_X = len(np.unique(bio_X))
            leg1 = ax.flat[i].hist(hist_dat,
                                   label=class_names,
                                   density=True,
                                   alpha=0.7,
                                   stacked=True)
            linspace = np.linspace(bio_X.min(), bio_X.max(), 100).reshape(-1, 1)
            controls_score = sp.stats.norm.pdf(linspace, loc=self.thetas[i][0], scale=self.thetas[i][1]) * self.thetas[i][4]
            patholog_score = sp.stats.norm.pdf(linspace, loc=self.thetas[i][2], scale=self.thetas[i][3]) * (1-self.thetas[i][4])
            ax.flat[i].plot(linspace, controls_score)
            ax.flat[i].plot(linspace, patholog_score)
            ax.flat[i].set_title(score_names[i])
            ax.flat[i].axes.get_yaxis().set_visible(False)


    def write(self, path='model.pkl'):
        file_out = Path(path)
        pickle_file = open(file_out, 'wb')
        data = {}
        data['params'] = self.params
        data['thetas'] = self.thetas
        pickle.dump(data, pickle_file)
        pickle_file.close()
    
    def read(self, path='model.pkl'):
        file_in = Path(path)
        pickle_file = open(file_in, 'rb')
        data = pickle.load(pickle_file)
        self.params = data['params']
        self.thetas = data['thetas']
        pickle_file.close()
        self.calc_prob_mat(self.X)
