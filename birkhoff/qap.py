from scipy.optimize import linear_sum_assignment
import numpy as np


## Helpers
def _trace_product(A, B, C, D):
    # return np.sum(np.trace(
    #     np.matmul(A, np.matmul(B, np.matmul(C, D))),
    #     axis1=1, axis2=2))
    AB = np.matmul(A, B)
    CD = np.matmul(C, D)
    return np.sum(AB * np.swapaxes(CD, -1, -2))



def _trace_product_linear(A, B):
    # return np.sum(np.trace(np.matmul(A, B), axis1=1, axis2=2))
    return np.sum(A * np.swapaxes(B, -1, -2))


### Inner steps of QAP solver
def _compute_gradient(P, As, Bs, Cs):
    g = np.matmul(np.swapaxes(As, 2, 1), np.matmul(P, Bs)).sum(axis=0)
    g += np.matmul(As, np.matmul(P, np.swapaxes(Bs, 2, 1))).sum(axis=0)
    g += Cs.sum(axis=0)
    return g


def _compute_step_size(P, Q, As, Bs, Cs):
    f0 = _trace_product(As, P, np.swapaxes(Bs, 2, 1), P.T)
    f1 = _trace_product(As, P, np.swapaxes(Bs, 2, 1), Q.T)
    f2 = _trace_product(As, Q, np.swapaxes(Bs, 2, 1), P.T)
    f3 = _trace_product(As, Q, np.swapaxes(Bs, 2, 1), Q.T)
    f4 = _trace_product_linear(np.swapaxes(Cs, 2, 1), P)
    f5 = _trace_product_linear(np.swapaxes(Cs, 2, 1), Q)

    num = 2 * f0 - f1 - f2 + f4 - f5
    denom = f0 + f3 - f1 - f2 + 1e-8
    a0 = min(1, max(0,  num / denom) / 2.0)
    a1 = 1
    a2 = 0
    avs = np.array([a0, a1, a2])

    costs = np.zeros(3)
    for i in range(3):
        PQ = (1 - avs[i]) * P + avs[i] * Q
        costs[i] = _trace_product(As, PQ, np.swapaxes(Bs, 2, 1), PQ.T) + \
                   _trace_product_linear(np.swapaxes(Cs, 2, 1), PQ)
    return avs[np.argmin(costs)]


### Approximate QAP solver
def solve_qap(As, Bs, Cs, P0=None, max_iter=100):
    """
    This code allows to solve the problem:

        \sum_i Tr(A_i P B_i^T P^T) + Tr(C_i^T P)

    subject to P being a permutation matrix. It is largely based on
    the paper: Fast Approximate Quadratic Programming for Graph Matching
    http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0121002

    :param As:  stack of A_i matrices   (D x N x N)
    :param Bs:  stack of B_i matrices   (D x N x N)
    :param Cs:  stack of C_i matrices   (D x N x N)
    :param P0:  initial permutation     (N x N)
    :param max_iter: max number of iterations
    :return: (N x N) permutation matrix that minimizes the functional above.
    """
    D, N, _ = As.shape
    assert As.shape == (D, N, N)
    assert Bs.shape == (D, N, N)
    assert Cs.shape == (D, N, N)

    # Initialize
    P = P0.copy() if P0 is not None else 1./N * np.ones((N, N))

    err = np.inf
    for itr in range(max_iter):
        if err < 1e-10:
            break

        # Compute the gradient
        G = _compute_gradient(P, As, Bs, Cs)

        # Find 'Q' by solving a linear cost assignment problem
        row_ind, col_ind = linear_sum_assignment(G)
        Q = np.zeros((N, N))
        Q[row_ind, col_ind] = 1
        
        # Compute step size 'alpha'
        alpha = _compute_step_size(P, Q, As, Bs, Cs)

        # Take a step of size alpha in the direction of the gradient
        P_new = (1.0 - alpha) * P + alpha * Q

        # Recompute error
        # err = alpha * np.linalg.norm(Q - P_new, ord='fro')
        err = np.linalg.norm(P_new - P, ord='fro')
        P = P_new

    # Project the final P onto the set of permutation matrices
    row_ind, col_ind = linear_sum_assignment(-P)
    P = np.zeros((N, N))
    P[row_ind, col_ind] = 1
    return P
