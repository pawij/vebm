"""
Forward mode differentiation for stick breaking transformations.
In the stick breaking transformations, all of the computations are
addition and subtraction of scalars.  This makes it pretty easy to
do the forward mode autodiff
"""
import autograd.numpy as np
from autograd.core import primitive, Node

#import birkhoff.cython_primitives as cython_primitives
from . import cython_primitives


### Helpers
def logistic(psi):
    return 1. / (1 + np.exp(-psi))

def logit(p):
    return np.log(p / (1-p))

def dlogit(p):
    return (1.0 / p) * (1.0 / (1.0 - p))

def log_dlogit(p):
    return -np.log(p) - np.log(1-p)

def gaussian_logp(x, mu, sigma):
    return np.sum(-0.5 * np.log(2 * np.pi * sigma**2) -0.5 * (x - mu)**2 / sigma**2)

def gaussian_entropy(log_sigma):
    return 0.5 * log_sigma.size * (1.0 + np.log(2 * np.pi)) + np.sum(log_sigma)

### 1D stick breaking for categoricals
def python_psi_to_pi(psi, return_intermediates=False):
    """
    Convert psi to a probability vector pi
    :param psi:     Length K-1 vector in [0,1]
    :return:        Length K normalized probability vector
    """
    K = psi.size + 1
    pi = np.zeros(K)

    # Intermediate terms
    ubs = np.zeros(K)
    ubs[0] = 1.0

    # Set pi[1..K-1]
    for k in range(K-1):
        pi[k] = psi[k] * ubs[k]
        ubs[k+1] = ubs[k] - pi[k]

    # Set the last output
    pi[K-1] = ubs[K-1]

    if return_intermediates:
        return pi, ubs
    else:
        return pi

def cython_psi_to_pi(psi):
    psi = psi.value if isinstance(psi, Node) else psi
    K = psi.shape[0] + 1
    pi = np.zeros(K)
    ubs = np.ones(K)
    cython_primitives.cython_psi_to_pi(psi, ubs, pi)
    return pi

def python_jacobian_psi_to_pi(psi):
    """
    J = [  -- dpi_1 / dpsi -- ]
        [  -- dpi_2 / dpsi -- ]
        [         ...         ]
        [  -- dpi_K / dpsi -- ]

    Output is K x K-1 matrix
    """
    # Initialize output
    K = psi.size + 1
    J = np.zeros((K, K-1))

    # Run once to get intermediate computations
    pi, ubs = python_psi_to_pi(psi, return_intermediates=True)

    for ii in range(K-1):
        dpsi = np.zeros(K - 1)
        dpsi[ii] = 1.0

        dpi = np.zeros(K)
        dubs = np.zeros(K)

        # Set pi[1..K-1]
        for k in range(K - 1):
            # Step 1: weight by ubs
            dpi[k] = dpsi[k] * ubs[k] + psi[k] * dubs[k]
            # Step 2: Adjust stick
            dubs[k + 1] = dubs[k] - dpi[k]

        # Set the last output
        dpi[K - 1] = dubs[K - 1]
        J[:, ii] = dpi
    return J

def cython_jacobian_psi_to_pi(psi):
    psi = psi.value if isinstance(psi, Node) else psi
    K = psi.shape[0] + 1
    pi = np.zeros(K)
    ubs = np.ones(K)
    J = np.zeros((K, K-1))
    cython_primitives.cython_jacobian_psi_to_pi(psi, ubs, pi, J)
    return J

psi_to_pi = primitive(cython_psi_to_pi)
psi_to_pi.defvjp(lambda g, ans, vs, gvs, psi, **kwargs:
                 np.dot(g, cython_jacobian_psi_to_pi(psi)))


### Stick breaking transformations for permutations
def python_psi_to_birkhoff(Psi, return_intermediates=False):
    """
    Transform a (K-1) x (K-1) matrix Psi into a KxK doubly
    stochastic matrix Pi.  I've introduced some tolerance to
    make it more stable, but I'm not convinced of this approach yet.

    Note that this version uses direct assignment to the matrix P,
    but this is not amenable to automatic differentiation with
    autograd. The next version below uses list comprehension to
    perform the same psi -> pi conversion in an automatically
    differentiable way.
    """
    K = Psi.shape[0] + 1
    assert Psi.shape == (K - 1, K - 1)

    # Initialize intermediate values
    P = np.zeros((K, K), dtype=float)
    ub_rows = np.ones((K, K))
    ub_cols = np.ones((K, K))
    lbs = np.zeros((K, K))

    for i in range(K - 1):
        for j in range(K - 1):
            # Compute lower bound
            lbs[i, j] = ub_rows[i, j] - ub_cols[i, j + 1:].sum()

            # Four cases
            if ub_rows[i, j] < ub_cols[i, j]:
                if lbs[i, j] > 0:
                    P[i, j] = lbs[i, j] + (ub_rows[i, j] - lbs[i, j]) * Psi[i, j]
                else:
                    P[i, j] = ub_rows[i, j] * Psi[i, j]
            else:
                if lbs[i, j] > 0:
                    P[i, j] = lbs[i, j] + (ub_cols[i, j] - lbs[i, j]) * Psi[i, j]
                else:
                    P[i, j] = ub_cols[i, j] * Psi[i, j]

            # Update upper bounds
            ub_rows[i, j+1] = ub_rows[i, j] - P[i, j]
            ub_cols[i+1, j] = ub_cols[i, j] - P[i, j]

        # Finish off the row
        P[i, -1] = ub_rows[i, -1]
        ub_cols[i+1, -1] = ub_cols[i, -1] - P[i, -1]

    # Finish off the columns
    for j in range(K-1):
        P[-1, j] = ub_cols[-1, j]

    # Compute the bottom right entry
    P[-1,-1] = 1 - np.sum(ub_cols[-1,:-1])

    if return_intermediates:
        return P, ub_rows, ub_cols, lbs
    else:
        return P

def cython_psi_to_birkhoff(Psi):
    Psi = Psi.value if isinstance(Psi, Node) else Psi
    K = Psi.shape[0] + 1
    P = np.zeros((K, K))
    ub_rows = np.ones((K, K))
    ub_cols = np.ones((K, K))
    lbs = np.zeros((K, K))
    cython_primitives.cython_psi_to_birkhoff(Psi, P, ub_rows, ub_cols, lbs)
    return P

def python_jacobian_psi_to_birkhoff(Psi):
    """
    As above, use dual numbers to perform forward mode
    differentiation.
    """
    K = Psi.shape[0] + 1
    assert Psi.shape == (K - 1, K - 1)

    # Run once to get the intermediate values
    P, ub_rows, ub_cols, lbs = python_psi_to_birkhoff(Psi, return_intermediates=True)

    # Initialize the Jacobian
    J = np.zeros((K, K, K-1, K-1))

    # Initialize intermediate values and derivatives
    for ii in range(K-1):
        for jj in range(K-1):
            dPsi = np.zeros((K - 1, K - 1))
            dPsi[ii, jj] = 1.0

            dP = np.zeros((K, K))
            dub_rows = np.zeros((K, K))
            dub_cols = np.zeros((K, K))
            dlbs = np.zeros((K, K))

            # Due to the feedforward nature, all the partial derivatives
            # are zero before the (ii,jj) output.
            for i in range(ii, K - 1):
                for j in range(jj, K - 1):
                    # Compute lower bound
                    dlbs[i, j] = dub_rows[i, j] - dub_cols[i,j+1:].sum()

                    # The upper bounds are computed recursively
                    if ub_rows[i, j] < ub_cols[i, j]:
                        if lbs[i, j] > 0:
                            dP[i, j] = dlbs[i, j] + \
                                       (dub_rows[i, j] - dlbs[i, j]) * Psi[i, j] + \
                                       (ub_rows[i, j] - lbs[i, j]) * dPsi[i, j]
                        else:
                            dP[i, j] = dub_rows[i, j] * Psi[i, j] + ub_rows[i, j] * dPsi[i, j]
                    else:
                        if lbs[i, j] > 0:
                            dP[i, j] = dlbs[i, j] + \
                                       (dub_cols[i, j] - dlbs[i, j]) * Psi[i, j] + \
                                       (ub_cols[i, j] - lbs[i, j]) * dPsi[i, j]
                        else:
                            dP[i, j] = dub_cols[i, j] * Psi[i, j] + ub_cols[i, j] * dPsi[i, j]


                    # Update upper bounds
                    dub_rows[i, j+1] = dub_rows[i, j] - dP[i, j]
                    dub_cols[i+1, j] = dub_cols[i, j] - dP[i, j]

                # Finish off the row
                dP[i, -1] = dub_rows[i, -1]
                dub_cols[i+1, -1] = dub_cols[i, -1] - dP[i, -1]

            # Finish off the columns
            for j in range(K-1):
                dP[-1, j] = dub_cols[-1, j]

            # Compute the bottom right entry
            dP[-1,-1] = -np.sum(dub_cols[-1,:-1])

            J[:,:,ii,jj] = dP

    return J

def cython_jacobian_psi_to_birkhoff(Psi):
    Psi = Psi.value if isinstance(Psi, Node) else Psi
    K = Psi.shape[0] + 1
    P = np.zeros((K, K))
    ub_rows = np.zeros((K, K))
    ub_cols = np.zeros((K, K))
    lbs = np.zeros((K, K))
    J = np.zeros((K, K, K-1, K-1))
    cython_primitives.cython_jacobian_psi_to_birkhoff(Psi, P, ub_rows, ub_cols, lbs, J)
    return J

psi_to_birkhoff = primitive(cython_psi_to_birkhoff)
psi_to_birkhoff.defvjp(lambda g, ans, vs, gvs, Psi, **kwargs:
                       np.tensordot(g, cython_jacobian_psi_to_birkhoff(Psi), ((0, 1), (0, 1))))

### Log determinants
def python_log_det_jacobian(P, return_intermediates=False):
    """
    Compute log det of the jacobian of the inverse transformation.
    Evaluate it at the given value of Pi.
    :param Pi: Doubly stochastic matrix
    :return: |dPsi / dPi |
    """
    K = P.shape[0]
    assert P.shape == (K, K)

    # Initialize output
    logdet = 0

    # Initialize intermediate values
    ub_rows = np.ones((K, K))
    ub_cols = np.ones((K, K))
    lbs = np.zeros((K, K))
    for i in range(K - 1):
        for j in range(K - 1):
            # Compute lower bound
            lbs[i, j] = ub_rows[i, j] - ub_cols[i, j + 1:].sum()

            # Four cases
            if ub_rows[i, j] < ub_cols[i, j]:
                if lbs[i, j] > 0:
                    logdet -= np.log(ub_rows[i, j] - lbs[i, j])
                else:
                    logdet -= np.log(ub_rows[i, j])
            else:
                if lbs[i, j] > 0:
                    logdet -= np.log(ub_cols[i, j] - lbs[i, j])
                else:
                    logdet -= np.log(ub_cols[i, j])

            # Update upper bounds
            ub_rows[i, j + 1] = ub_rows[i, j] - P[i, j]
            ub_cols[i + 1, j] = ub_cols[i, j] - P[i, j]

        # Finish off the row
        ub_cols[i + 1, -1] = ub_cols[i, -1] - P[i, -1]

    if return_intermediates:
        return logdet, ub_rows, ub_cols, lbs
    else:
        return logdet

def cython_log_det_jacobian(P):
    P = P.value if isinstance(P, Node) else P
    K = P.shape[0]
    ub_rows = np.zeros((K, K))
    ub_cols = np.zeros((K, K))
    lbs = np.zeros((K, K))
    return cython_primitives.cython_log_det_jacobian(P, ub_rows, ub_cols, lbs)

def python_grad_log_det_jacobian(P):
    """
    Gradient of the log det calculation.
    """
    K = P.shape[0]
    assert P.shape == (K, K)

    # Initialize output
    dlogdet = np.zeros((K, K))

    # Call once to get intermediates
    _, ub_rows, ub_cols, lbs = python_log_det_jacobian(P, return_intermediates=True)

    for ii in range(K):
        for jj in range(K):
            # Initialize input
            dP = np.zeros((K, K))
            dP[ii, jj] = 1

            # Initialize intermediate values
            dub_rows = np.zeros((K, K))
            dub_cols = np.zeros((K, K))
            dlbs = np.zeros((K, K))

            for i in range(K - 1):
                for j in range(K - 1):
                    # Compute lower bound
                    dlbs[i, j] = dub_rows[i, j] - dub_cols[i, j + 1:].sum()

                    # Four cases
                    if ub_rows[i, j] < ub_cols[i, j]:
                        if lbs[i, j] > 0:
                            dlogdet[ii, jj] -= (dub_rows[i, j] - dlbs[i, j]) / \
                                               (ub_rows[i, j] - lbs[i, j])

                        else:
                            dlogdet[ii, jj] -= dub_rows[i, j] / ub_rows[i, j]
                    else:
                        if lbs[i, j] > 0:
                            dlogdet[ii, jj] -= (dub_cols[i, j] - dlbs[i, j]) / \
                                               (ub_cols[i, j] - lbs[i, j])
                        else:
                            dlogdet[ii, jj] -= dub_cols[i, j] / ub_cols[i, j]

                    # Update upper bounds
                    dub_rows[i, j + 1] = dub_rows[i, j] - dP[i, j]
                    dub_cols[i + 1, j] = dub_cols[i, j] - dP[i, j]

                # Finish off the row
                dub_cols[i + 1, -1] = dub_cols[i, -1] - dP[i, -1]

    return dlogdet

def cython_grad_log_det_jacobian(P):
    P = P.value if isinstance(P, Node) else P
    K = P.shape[0]
    ub_rows = np.zeros((K, K))
    ub_cols = np.zeros((K, K))
    lbs = np.zeros((K, K))
    dlogdet = np.zeros((K, K))
    cython_primitives.cython_grad_log_det_jacobian(P, ub_rows, ub_cols, lbs, dlogdet)
    return dlogdet

log_det_jacobian = primitive(cython_log_det_jacobian)
log_det_jacobian.defvjp(lambda g, ans, vs, gvs, P, **kwargs:
                        np.full(P.shape, g) * cython_grad_log_det_jacobian(P))

### Invert the transformation
def birkhoff_to_psi(P, verbose=False):
    """
    Invert Pi to get Psi, assuming Pi was sampled using
    sample_doubly_stochastic_stable() with the same tolerance.
    """
    N = P.shape[0]
    assert P.shape == (N, N)

    # Psi = np.zeros((N - 1, N - 1), dtype=float)
    Psi = []
    for i in range(N - 1):
        Psi_i = []
        for j in range(N - 1):
            # Upper bounded by partial row sum
            # Upper bounded by partial column sum
            ub_row = 1 - np.sum(P[i, :j])
            ub_col = 1 - np.sum(P[:i, j])

            # Lower bounded (see notes)
            lb_rem = (1 - np.sum(P[i, :j])) - (N - (j + 1)) + np.sum(P[:i, j + 1:])

            # Combine constraints
            ub = min(ub_row, ub_col)
            lb = max(0, lb_rem)

            if verbose:
                print("({0}, {1}): lb_rem: {2} lb: {3}  ub: {4}".format(i, j, lb_rem, lb, ub))

            # Check if difference is less than allowable tolerance
            Psi_i.append((P[i, j] - lb) / (ub - lb))

        Psi.append(Psi_i)
    return np.array(Psi)

def log_density_pi(P, mu, sigma, verbose=False):
    """
    Compute the log probability of a permutation matrix for a given
    mu and sigma---the parameters of the Gaussian prior on psi.
    """
    N = P.shape[0]
    assert P.shape == (N, N)
    assert mu.shape == (N-1, N-1)
    assert sigma.shape == (N-1, N-1)

    Psi = birkhoff_to_psi(P[:-1, :-1], verbose=verbose)
    Psi = logit(Psi)
    return log_det_jacobian(P) + \
           np.sum(gaussian_logp(Psi, mu, sigma))

def birkhoff_to_perm(B):
    from scipy.optimize import linear_sum_assignment
    K = B.shape[0]
    assert B.shape == (K, K)
    row, col = linear_sum_assignment(-B)
    P = np.zeros((K, K))
    P[row, col] = 1.0
    return P
