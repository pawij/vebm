import pickle
import os
import numpy as np
import scipy
from scipy.misc import logsumexp
from scipy.special import gammaln, beta
from scipy.integrate import simps

import scipy.sparse

def logistic(x):
    return 1./(1+np.exp(-x))

def logit(p):
    return np.log(p/(1-p))

def psi_to_pi(psi, axis=None):
    """
    Convert psi to a probability vector pi
    :param psi:     Length K-1 vector
    :return:        Length K normalized probability vector
    """
    if axis is None:
        if psi.ndim == 1:
            K = psi.size + 1
            pi = np.zeros(K)

            # Set pi[1..K-1]
            stick = 1.0
            for k in range(K-1):
                pi[k] = logistic(psi[k]) * stick
                stick -= pi[k]

            # Set the last output
            pi[-1] = stick
            # DEBUG
            assert np.allclose(pi.sum(), 1.0)

        elif psi.ndim == 2:
            M, Km1 = psi.shape
            K = Km1 + 1
            pi = np.zeros((M,K))

            # Set pi[1..K-1]
            stick = np.ones(M)
            for k in range(K-1):
                pi[:,k] = logistic(psi[:,k]) * stick
                stick -= pi[:,k]

            # Set the last output
            pi[:,-1] = stick

            # DEBUG
            assert np.allclose(pi.sum(axis=1), 1.0)

        else:
            raise ValueError("psi must be 1 or 2D")
    else:
        K = psi.shape[axis] + 1
        pi = np.zeros([psi.shape[dim] if dim != axis else K for dim in range(psi.ndim)])
        stick = np.ones(psi.shape[:axis] + psi.shape[axis+1:])
        for k in range(K-1):
            inds = [slice(None) if dim != axis else k for dim in range(psi.ndim)]
            pi[inds] = logistic(psi[inds]) * stick
            stick -= pi[inds]
        pi[[slice(None) if dim != axis else -1 for dim in range(psi.ndim)]] = stick
        assert np.allclose(pi.sum(axis=axis), 1.)

    return pi

def pi_to_psi(pi):
    """
    Convert probability vector pi to a vector psi
    :param pi:      Length K probability vector
    :return:        Length K-1 transformed vector psi
    """
    if pi.ndim == 1:
        K = pi.size
        assert np.allclose(pi.sum(), 1.0)
        psi = np.zeros(K-1)

        stick = 1.0
        for k in range(K-1):
            psi[k] = logit(pi[k] / stick)
            stick -= pi[k]

        # DEBUG
        assert np.allclose(stick, pi[-1])
    elif pi.ndim == 2:
        M, K = pi.shape
        assert np.allclose(pi.sum(axis=1), 1.0)
        psi = np.zeros((M,K-1))

        stick = np.ones(M)
        for k in range(K-1):
            psi[:,k] = logit(pi[:,k] / stick)
            stick -= pi[:,k]
        assert np.allclose(stick, pi[:,-1])
    else:
        raise NotImplementedError

    return psi

def det_jacobian_pi_to_psi(pi):
    """
    Compute |J| = |d\psi_j / d\pi_k| = the jacobian of the mapping from
     pi to psi. Since we have a stick breaking construction, the Jacobian
     is lower triangular and the determinant is simply the product of the
     diagonal. For our model, this can be computed in closed form. See the
     appendix of the draft.

    :param pi: K dimensional probability vector
    :return:
    """
    # import pdb; pdb.set_trace()
    K = pi.size

    # Jacobian is K-1 x K-1
    diag = np.zeros(K-1)
    for k in range(K-1):
        diag[k] = (1.0 - pi[:k].sum()) / (pi[k] * (1-pi[:(k+1)].sum()))

    det_jacobian = diag.prod()
    return det_jacobian

def det_jacobian_psi_to_pi(psi):
    """
    Compute the Jacobian of the inverse mapping psi to pi.
    :param psi:
    :return:
    """
    pi = psi_to_pi(psi)
    return 1.0 / det_jacobian_pi_to_psi(pi)


def gaussian_to_pi_density(psi_mesh, mu, Sigma):
    pi_mesh = np.array(list(map(psi_to_pi, psi_mesh)))
    valid_pi = np.all(np.isfinite(pi_mesh), axis=1)
    pi_mesh = pi_mesh[valid_pi,:]

    # Compute the det of the Jacobian of the inverse mapping
    det_jacobian = np.array(list(map(det_jacobian_pi_to_psi, pi_mesh)))
    det_jacobian = det_jacobian[valid_pi]

    # Compute the multivariate Gaussian density
    # pi_pdf = np.exp(log_dirichlet_density(pi_mesh, alpha=alpha))
    from scipy.stats import multivariate_normal
    psi_dist = multivariate_normal(mu, Sigma)
    psi_pdf = psi_dist.pdf(psi_mesh)
    psi_pdf = psi_pdf[valid_pi]

    # The psi density is scaled by the det of the Jacobian
    pi_pdf = psi_pdf * det_jacobian

    return pi_mesh, pi_pdf

def ln_psi_to_pi(psi):
    """
    Convert the logistic normal psi to a probability vector pi
    :param psi:     Length K vector
    :return:        Length K normalized probability vector
    """
    lognumer = psi

    if psi.ndim == 1:
        logdenom = logsumexp(psi)
    elif psi.ndim == 2:
        logdenom = logsumexp(psi, axis=1)[:, None]
    pi = np.exp(lognumer - logdenom)
    # assert np.allclose(pi.sum(), 1.0)

    return pi

def ln_pi_to_psi(pi, scale=1.0):
    """
    Convert the logistic normal psi to a probability vector pi
    The transformation from psi to pi is not invertible unless
    you know the scaling of the psis.

    :param pi:      Length K vector
    :return:        Length K unnormalized real vector
    """
    assert scale > 0
    if pi.ndim == 1:
        assert np.allclose(pi.sum(), 1.0)
    elif pi.ndim == 2:
        assert np.allclose(pi.sum(1), 1.0)

    psi = np.log(pi) + np.log(scale)

    # assert np.allclose(pi, ln_psi_to_pi(psi))
    return psi

def compute_uniform_mean_psi(K, alpha=2):
    """
    Compute the multivariate distribution over psi that will yield approximately
    Dirichlet(\alpha) prior over pi

    :param K:   Number of entries in pi
    :return:    A K-1 vector mu that yields approximately uniform distribution over pi
    """
    mu, sigma = compute_psi_cmoments(alpha*np.ones(K))
    return mu, np.diag(sigma)

def compute_psi_cmoments(alphas):
    K = alphas.shape[0]
    psi = np.linspace(-10,10,1000)

    mu = np.zeros(K-1)
    sigma = np.zeros(K-1)
    for k in range(K-1):
        density = get_density(alphas[k], alphas[k+1:].sum())
        mu[k] = simps(psi*density(psi),psi)
        sigma[k] = simps(psi**2*density(psi),psi) - mu[k]**2
        # print '%d: mean=%0.3f var=%0.3f' % (k, mean, s - mean**2)

    return mu, sigma

def get_density(alpha_k, alpha_rest):
    def density(psi):
        return logistic(psi)**alpha_k * logistic(-psi)**alpha_rest \
            / scipy.special.beta(alpha_k,alpha_rest)
    return density

def plot_psi_marginals(alphas):
    K = alphas.shape[0]
    psi = np.linspace(-10,10,1000)

    import matplotlib.pyplot as plt
    plt.figure()

    for k in range(K-1):
        density = get_density(alphas[k], alphas[k+1:].sum())
        plt.subplot(2,1,1)
        plt.plot(psi,density(psi),label='psi_%d' % k)
        plt.subplot(2,1,2)
        plt.plot(psi,np.log(density(psi)),label='psi_%d' % k)
    plt.subplot(2,1,1)
    plt.legend()

def N_vec(x, axis=None):
    """
    Compute the count vector for PG Multinomial inference
    :param x:
    :return:
    """
    if axis is None:
        if x.ndim == 1:
            N = x.sum()
            return np.concatenate(([N], N - np.cumsum(x)[:-2]))
        elif x.ndim == 2:
            N = x.sum(axis=1)
            return np.hstack((N[:,None], N[:,None] - np.cumsum(x, axis=1)[:,:-2]))
        else:
            raise ValueError("x must be 1 or 2D")
    else:
        inds = [slice(None) if dim != axis else None for dim in range(x.ndim)]
        inds2 = [slice(None) if dim != axis else slice(None,-2) for dim in range(x.ndim)]
        N = x.sum(axis=axis)
        return np.concatenate((N[inds], N[inds] - np.cumsum(x,axis=axis)[inds2]), axis=axis)

def kappa_vec(x, axis=None):
    """
    Compute the kappa vector for PG Multinomial inference
    :param x:
    :return:
    """
    if axis is None:
        if x.ndim == 1:
            return x[:-1] - N_vec(x)/2.0
        elif x.ndim == 2:
            return x[:,:-1] - N_vec(x)/2.0
        else:
            raise ValueError("x must be 1 or 2D")
    else:
        inds = [slice(None) if dim != axis else slice(None,-1) for dim in range(x.ndim)]
        return x[inds] - N_vec(x, axis)/2.0

# is this doing overlapping work with dirichlet_to_psi_density_closed_form?
def get_marginal_psi_density(alpha_k, alpha_rest):
    def density(psi):
        return logistic(psi)**alpha_k * logistic(-psi)**alpha_rest \
            / beta(alpha_k,alpha_rest)
    return density


def dirichlet_to_psi_meanvar(alphas,psigrid=np.linspace(-10,10,1000)):
    K = alphas.shape[0]

    def meanvar(k):
        density = get_marginal_psi_density(alphas[k], alphas[k+1:].sum())
        mean = simps(psigrid*density(psigrid),psigrid)
        s = simps(psigrid**2*density(psigrid),psigrid)
        return mean, s - mean**2

    return list(map(np.array, list(zip(*[meanvar(k) for k in range(K-1)]))))


def sinkhorn(P, n_iter=100):
    """
    `Project' a nonnegative matrix P onto the Birkhoff polytope
    by iterative row/column normalization.
    """
    K = P.shape[0]
    assert P.shape == (K,K)
    P_ds = P.copy()
    for itr in range(n_iter):
        P_ds /= P_ds.sum(axis=1)[:,None]
        P_ds /= P_ds.sum(axis=0)[None,:]
    return P_ds


### Construct a low dimensional parameterization
def get_birkhoff_basis(K):
    # Construct a basis for the orthogonal complement
    # of the all-ones rows and all-ones columns
    Ws = []
    for n in range(K):
        # row
        Wnr = np.zeros((K, K))
        Wnr[n, :] = 1
        Ws.append(Wnr.ravel())

        # column
        Wnc = np.zeros((K, K))
        Wnc[:, n] = 1
        Ws.append(Wnc.ravel())
    Ws = np.column_stack(Ws)

    # We want the orthogonal complement of the columns of Ws
    U, R = np.linalg.qr(Ws, mode='complete')

    # Ws.shape = (N^2 x 2N) so U.shape = (N^2 x N^2)
    # We expect the rank of U to be (N-1)^2 since that
    # is the dimension of the hypersphere.
    #
    # print("bases of Ws")
    # for i in range(2 * K - 1):
    #     ui = U[:, i]
    #     print(abs(ui.dot(Ws)).sum())
    #
    # print("")
    # print("bases of \perp(Ws)")
    # for i in range(2 * K - 1, K ** 2):
    #     ui = U[:, i]
    #     print(ui.dot(Ws).sum())

    # Get the basis of the complement
    # Shape = N^2 x (N-1)^2
    U_compl = U[:, 2 * K - 1:]
    assert U_compl.shape == (K ** 2, (K - 1) ** 2)
    return U_compl

# Sweeeeeeeet... now let's project permutations onto the sphere
def project_perm_to_sphere(P, U_compl):
    N = P.shape[0]
    assert P.shape == (N, N)

    # Vectorize, center, project, reshape
    P = P.ravel()
    P = P - 1.0 / N * np.ones(N ** 2)
    Psi = U_compl.T.dot(P)
    return Psi


def project_sphere_to_perm(Psi, U_compl):
    assert Psi.ndim == 1
    K = np.sqrt(Psi.size) + 1
    assert np.allclose(K % 1.0, 0.0)

    P = U_compl.dot(Psi)
    P = P + 1.0 / K * np.ones(K ** 2)
    P = P.reshape((K, K))
    return P


def random_permutation(K):
    perm = np.random.permutation(K)
    P = np.zeros((K, K))
    P[np.arange(K), perm] = 1
    return P


# Get a projection matrix
def get_birkhoff_projection(K):
    # Find a projection of permutations onto 2
    N = np.prod(np.arange(1, K+1))
    D_in = K**2
    D_out = 2
    ths = np.linspace(0, 2*np.pi, N, endpoint=False)
    Y = np.column_stack((np.cos(ths), np.sin(ths)))

    # Compute set of permutation matrices
    import itertools as it
    Ps = []
    for perm in it.permutations(np.arange(K)):
        P = np.zeros((K, K))
        P[np.arange(K), np.array(perm)] = 1
        Ps.append(P)
    Ps = np.array(Ps)
    X = Ps.reshape((N, D_in))

    # Solve least squares
    Q = np.linalg.solve(X.T.dot(X) + 0.1 * np.eye(D_in), X.T.dot(Y))
    Q, _ = np.linalg.qr(Q)
    return Q

def get_b3_projection():
    # Let's set the points in 2D and then try to figure out the projection
    K = 3
    Kfac = np.prod(np.arange(1, K + 1))
    ths = np.linspace(0, 2 * np.pi, Kfac, endpoint=False)
    Y = np.column_stack((np.cos(ths), np.sin(ths)))

    perms = [(2, 1, 0),
             (1, 2, 0),
             (0, 2, 1),
             (0, 1, 2),
             (1, 0, 2),
             (2, 0, 1)]

    Ps = []
    for perm in perms:
        P = np.zeros((K, K))
        P[np.arange(K), np.array(perm)] = 1
        Ps.append(P)
    Ps = np.array(Ps)
    X = Ps.reshape((Kfac, K ** 2))

    # Solve least squares
    Q = np.linalg.solve(X.T.dot(X) + 0.1 * np.eye(K**2), X.T.dot(Y))
    Q, _ = np.linalg.qr(Q)

    return Q

def perm_matrix(perm):
    K = len(perm)
    P = np.zeros((K, K))
    P[np.arange(K), perm] = 1
    assert np.all(P.sum(0) == 1)
    assert np.all(P.sum(1) == 1)
    return P

def check_doubly_stochastic(P):
    K = P.shape[0]
    assert P.shape == (K, K)
    assert np.all(P >= -1e-8)
    assert np.all(P <= 1+1e-8)
    assert np.allclose(P.sum(0), 1.0)
    assert np.allclose(P.sum(1), 1.0)

def cached(results_dir, results_name):
    def _cache(func):

        def func_wrapper(*args, **kwargs):
            results_file = os.path.join(results_dir, results_name)
            if not results_file.endswith(".pkl"):
                results_file += ".pkl"

            if os.path.exists(results_file):
                with open(results_file, "rb") as f:
                    results = pickle.load(f)

            else:
                results = func(*args, **kwargs)
                with open(results_file, "w+") as f:

                    pickle.dump(results, f)

            return results
        return func_wrapper
    return _cache

