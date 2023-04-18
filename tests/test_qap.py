import numpy as np
import numpy.random as npr
from scipy.optimize import linear_sum_assignment

from birkhoff.qap import solve_qap


def test_linear_assignment():
    # Simple test: linear cost sum assignment
    N = 10
    A = np.zeros((1, N, N))
    B = np.zeros((1, N, N))
    C = np.random.randn(1, N, N)
    P_initial = 0.1 / N * np.ones((N, N)) + 0.9 * np.eye(N)

    # Solve with approximate algorithm
    P_hat = solve_qap(A, B, C, P0=P_initial)

    # Solve using Hungarian algorithm
    row, col = linear_sum_assignment(C[0])
    P_test = np.zeros((N, N))
    P_test[row, col] = 1

    # Compare
    print(np.where(P_hat)[1])
    print(np.where(P_test)[1])
    assert(np.allclose(P_hat, P_test))


def test_quadratic_assignment(T=1000, N=100, etasq=0.1, rho=1.0):

    # Sample global dynamics matrix
    A = npr.rand(N, N) < rho
    W = np.sqrt(1./(2*N)) * npr.randn(N, N)
    assert np.all(abs(np.linalg.eigvals(A * W)) <= 1.0)

    # Sample permutations for each worm
    # perm[i] = index of neuron i in worm m's neurons
    perm = npr.permutation(N)
    P = np.zeros((N, N))
    P[np.arange(N), perm] = 1

    # Sample some data!
    Y = np.zeros((T, N))
    Y[0, :] = np.ones(N)
    Wm = P.T.dot((W * A).dot(P))
    for t in range(1, T):
        mu_mt = np.dot(Wm, Y[t - 1, :])
        Y[t, :] = mu_mt + np.sqrt(etasq) * npr.randn(N)

    # Solve for the permutation that best explains the observed data Y
    # P | Y, W should is a quadratic assignment problem. Our cost is:
    #
    #   \sum_t (y[t] - P.T W P y[t-1])^2
    # = \sum_t -2 y[t] P.T W P y[t-1] + y[t-1].T P.T W.T P P.T W P y[t]
    # = \sum_t -2 y[t].T P.T W P y[t-1] + y[t-1].T P.T W.T W P y[t-1]
    #
    # = Tr(- 2 (\sum_t y[t-1] y[t].T) P.T W P)
    #   + Tr( (\sum_t y[t-1] y[t-1].T) P.T W.T W P)
    #
    yp, y = Y[:-1], Y[1:]
    A1 = -2 * np.sum(yp[:, None, :] * y[:, :, None], axis=0)
    B1 = (W * A)
    A2 = np.sum(yp[:, None, :] * yp[:, :, None], axis=0)
    B2 = np.dot((W * A).T, (W * A))
    C1 = C2 = np.zeros((N, N))

    P_hat = solve_qap(np.array([A1, A2]),
                      np.array([B1, B2]),
                      np.array([C1, C2]))

    # Note that solve_qap uses actually yields the transpose of P
    P_hat = P_hat.T

    print(np.where(P)[1])
    print(np.where(P_hat)[1])

if __name__ == "__main__":
    # test_linear_assignment()
    test_quadratic_assignment()
