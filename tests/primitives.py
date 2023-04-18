import autograd.numpy as np
from autograd import jacobian, grad
from birkhoff.utils import psi_to_pi as psi_to_pi_old
from birkhoff.utils import logistic

from birkhoff.primitives import *

### Simple tests
def _psi_to_pi_autograd(psi):
    pi = []
    for i in range(psi.shape[0]):
        pi.append(psi[i] * (1 - np.sum(np.array(pi),0)))
    pi.append(1-np.sum(np.array(pi)))
    return np.array(pi)

def _psi_to_birkhoff_autograd(Psi, verbose=False):
    """
    Autograd doesn't work with array assignment.  Rewrite with
    list comprehension instead.  This will be a bit slower, but
    not too bad.
    """
    N = Psi.shape[0] + 1
    assert Psi.shape == (N - 1, N - 1)

    # P = np.zeros((N, N), dtype=float)
    P = []
    for i in range(N - 1):
        P_i = []
        for j in range(N - 1):
            # Upper bounded by partial row sum
            ub_row = 1.0
            for jj in range(j):
                ub_row = ub_row - P_i[jj]

            # Upper bounded by partial column sum
            # ub_col = 1 - np.sum(P[:i, j])
            ub_col = 1.0
            for ii in range(i):
                ub_col = ub_col - P[ii][j]

            # Lower bounded (see notes)
            # lb_rem = (1 - np.sum(P[i, :j])) - (N - (j + 1)) + np.sum(P[:i, j + 1:])
            lb_rem = 1.0 - (N - (j + 1))
            for jj in range(j):
                lb_rem = lb_rem - P_i[jj]

            for ii in range(i):
                for jj in range(j+1,N):
                    lb_rem = lb_rem + P[ii][jj]

            # Combine constraints
            ub = min(ub_row, ub_col)
            lb = max(0, lb_rem)

            if verbose:
                print("({0}, {1}): lb_rem: {2} lb: {3}  ub: {4}".format(i, j, lb_rem, lb, ub))

            # P[i, j] = lb + (ub - lb) * logistic(Psi[i, j])
            P_i.append(lb + (ub - lb) * Psi[i, j])

        # Finish off the row
        # P[i, -1] = 1 - np.sum(P[i, :-1])
        PiN = 1.0
        for jj in range(N-1):
            PiN = PiN - P_i[jj]
        P_i.append(PiN)

        # Append
        P.append(P_i)

    # Finish off the columns
    P_N = []
    for j in range(N):
        PjN = 1.0
        for ii in range(N-1):
            PjN = PjN - P[ii][j]
        P_N.append(PjN)
    P.append(P_N)

    return np.array(P)

def _log_det_jacobian_autograd(P, verbose=False):
    """
    Compute log det of the jacobian of the inverse transformation.
    Evaluate it at the given value of Pi.
    :param P: Given permutation matrix
    :return: |dPsi / dPi |
    """
    N = P.shape[0]
    assert P.shape == (N, N)

    logdet = 0
    for i in range(N - 1):
        for j in range(N - 1):
            # Upper bounded by partial row sum
            ub_row = 1 - np.sum(P[i, :j])
            # Upper bounded by partial column sum
            ub_col = 1 - np.sum(P[:i, j])

            # Lower bounded (see notes)
            lb_rem = (1 - np.sum(P[i, :j])) - (N - (j + 1)) + np.sum(P[:i, j + 1:])

            # Combine constraints
            ub = min(ub_row, ub_col)
            lb = max(0, lb_rem)

            if verbose:
                print("({0}, {1}): lb_rem: {2} lb: {3}  ub: {4}".format(i, j, lb_rem, lb, ub))

            # Check if difference is less than allowable tolerance
            # logdet = logdet + log_dlogit((P[i, j] - lb) / (ub - lb))
            logdet = logdet - np.log(ub - lb)

    return logdet

def test_psi_to_pi():
    K = 10
    psi = np.random.randn(K-1)
    pi1 = psi_to_pi_old(psi)
    pi2 = _psi_to_pi_autograd(logistic(psi))
    pi3 = python_psi_to_pi(logistic(psi))
    pi4 = psi_to_pi(logistic(psi))
    assert np.allclose(pi1, pi2)
    assert np.allclose(pi1, pi3)
    assert np.allclose(pi1, pi4)

def test_jacobian_psi_to_pi():
    K = 10
    psi = np.random.randn(K-1)
    p = logistic(psi)
    J1 = jacobian(_psi_to_pi_autograd)(p)
    J2 = jacobian(psi_to_pi)(p)
    J3 = python_jacobian_psi_to_pi(p)
    assert np.allclose(J1, J2)
    assert np.allclose(J1, J3)

def test_psi_to_birkhoff():
    """
    Make sure everything runs
    """
    K = 10
    mu = np.zeros((K - 1, K - 1))
    sigma = 0.1 * np.ones((K - 1, K - 1))
    Psi = mu + sigma * np.random.randn(K - 1, K - 1)

    P1 = _psi_to_birkhoff_autograd(logistic(Psi))
    P2 = psi_to_birkhoff(logistic(Psi))
    P3 = cython_psi_to_birkhoff(logistic(Psi))
    P4 = python_psi_to_birkhoff(logistic(Psi))

    assert np.allclose(P1, P2)
    assert np.allclose(P1, P3)
    assert np.allclose(P1, P4)


def test_jacobian_psi_to_birkhoff():
    K = 10
    Psi = np.random.randn(K-1, K-1)
    print("ag start")
    J1 = jacobian(_psi_to_birkhoff_autograd)(logistic(Psi))
    print("ag stop")
    print(J1.shape)

    print("manual start")
    J2 = jacobian(psi_to_birkhoff)(logistic(Psi))
    print("manual stop")

    print("manual start")
    J3 = python_jacobian_psi_to_birkhoff(logistic(Psi))
    print("manual stop")

    print("cython start")
    J4 = cython_jacobian_psi_to_birkhoff(logistic(Psi))
    print("cython stop")


    assert np.allclose(J2, J3)
    assert np.allclose(J1, J2)
    assert np.allclose(J1, J3)
    assert np.allclose(J1, J4)


def test_plot_jacobian():
    K = 4
    Psi = np.random.randn(K - 1, K - 1)
    J1 = jacobian(_psi_to_birkhoff_autograd)(logistic(Psi))
    J2 = cython_jacobian_psi_to_birkhoff(logistic(Psi))

    J1 = J1.reshape((K ** 2, (K - 1) ** 2))
    J2 = J2.reshape((K ** 2, (K - 1) ** 2))
    lim = max(abs(J1).max(), abs(J2).max())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.imshow(J1, interpolation="none", vmin=-lim, vmax=lim)
    plt.xlabel("$\\mathrm{vec}(\\Psi)$")
    plt.ylabel("$\\mathrm{vec}(\\Pi)$")
    plt.title("Lower triangular Jacobian")
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(J2, interpolation="none", vmin=-lim, vmax=lim)
    plt.xlabel("$\\mathrm{vec}(\\Psi)$")
    plt.ylabel("$\\mathrm{vec}(\\Pi)$")
    plt.title("Lower triangular Jacobian")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def test_log_det_jacobian():
    K = 4
    Psi = np.random.randn(K - 1, K - 1)
    P = python_psi_to_birkhoff(logistic(Psi))

    logdet1 = _log_det_jacobian_autograd(P)
    logdet2 = python_log_det_jacobian(P)
    logdet3 = cython_log_det_jacobian(P)

    # Manually compute the log det of the jacobian
    J = python_jacobian_psi_to_birkhoff(logistic(Psi))
    J = J[:-1, :-1, :, :].reshape(((K-1)**2, (K-1)**2))
    _, logdet4 = np.linalg.slogdet(J)

    assert np.allclose(logdet1, logdet2)
    assert np.allclose(logdet1, logdet3)
    assert np.allclose(logdet1, -logdet4)

def test_grad_log_det_jacobian():
    K = 4
    Psi = np.random.randn(K - 1, K - 1)
    P = python_psi_to_birkhoff(logistic(Psi))

    glogdet1 = grad(_log_det_jacobian_autograd)(P)
    glogdet2 = python_grad_log_det_jacobian(P)
    glogdet3 = cython_grad_log_det_jacobian(P)
    glogdet4 = grad(log_det_jacobian)(P)

    assert np.allclose(glogdet1, glogdet2)
    assert np.allclose(glogdet1, glogdet3)
    assert np.allclose(glogdet1, glogdet4)
    print(glogdet3)

if __name__ == "__main__":
    test_psi_to_pi()
    test_jacobian_psi_to_pi()
    test_psi_to_birkhoff()
    test_jacobian_psi_to_birkhoff()
    test_log_det_jacobian()
    test_grad_log_det_jacobian()
    # test_plot_jacobian()