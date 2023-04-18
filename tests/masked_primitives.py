import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import jacobian, grad
from birkhoff.utils import logistic, perm_matrix, check_doubly_stochastic

import birkhoff.primitives as primitives
import birkhoff.mask_primitives as masked

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")

### Simple tests
def random_permutation(K):
    perm = npr.permutation(K)
    return perm_matrix(perm)

def random_mask(K, per_row):
    # Sample a random permutation
    P = random_permutation(K)

    # Define a mask for possible assignments.
    M = np.ones((K, K), dtype=bool)
    for k in range(K):
        M[k, npr.choice(np.where(P[k] == 0)[0], K - per_row, replace=False)] = 0

    return M

def diagonal_mask(K, per_row=3):
    M = np.zeros((K, K), dtype=bool)
    for k in range((per_row+1)//2):
        M += np.eye(K, k=k).astype(bool)
        M += np.eye(K, k=-k).astype(bool)
    return M

def border_mask(K, per_row):
    # Sample a random permutation
    M = random_mask(K, per_row)
    M[:,-1] = 1
    M[-1,:] = 1
    return M

def test_psi_to_birkhoff_maksed1():
    """
    Make sure everything runs
    """
    from birkhoff.primitives import \
        logit, logistic, gaussian_logp, gaussian_entropy, \
        psi_to_birkhoff, log_det_jacobian, birkhoff_to_psi, \
        birkhoff_to_perm

    from birkhoff.mask_primitives import order_from_mask, python_psi_to_birkhoff_masked, NOT_DETERMINED

    # Global parameters
    npr.seed(0)
    K = 5
    D = 2
    eta = 0.5
    # Sample a true permutation (in=col, out=row)
    # perm_true = npr.permutation(K)
    perm_true = np.arange(K)[::-1]
    P_true = np.zeros((K, K))
    P_true[np.arange(K), perm_true] = 1

    # Sample data according to this permutation
    xs = npr.randn(K, D)
    xs_perm = P_true.T.dot(xs)
    ys = xs_perm + eta * npr.randn(K, D)

    # Define a mask for possible assignments.
    # M[i,j] = 0 implies P[i,j] = 0
    choices_per_node = 3
    M = np.ones((K, K), dtype=bool)
    for k in range(K):
        M[k, npr.choice(np.where(P_true[k] == 0)[0], K - choices_per_node, replace=False)] = False

    # Count the number of possible assignments
    print("Num possible assignments: ", M.sum())

    M = M.astype(bool)

    # Fingers crossed!
    N, rows, cols, determined = order_from_mask(M)

    M_order = np.zeros((K, K))
    M_determined = -1 * np.ones((K, K))
    for o, (i, j, d) in enumerate(zip(rows, cols, determined)):
        M_order[i, j] = 1 + o
        M_determined[i, j] = d

    plt.figure()
    plt.subplot(121)
    plt.imshow(M_order, interpolation="nearest")
    plt.subplot(122)
    plt.imshow(M_determined, interpolation="nearest")

    print(M_order)

    plt.show()


def test_psi_to_birkhoff_maksed2():
    """
    Make sure everything runs
    """
    K = 10
    M = np.eye(K, dtype=bool) | (npr.rand(K, K) < 0.8)
    for k in range(K):
        M[k,k+2:] = False

    # M = border_mask(K, 3)
    # M = diagonal_mask(K, 5)
    # M -= np.eye(K, k=-1).astype(bool)
    # M = np.fliplr(M)
    # M = np.ones((K, K), dtype=bool)
    preprocessed_mask = masked.preprocess_mask(M)
    K, N, N_inner, rows, cols, inner, rowend, colend = preprocessed_mask

    mu = np.zeros(N_inner)
    sigma = np.ones(N_inner)
    z = mu + sigma * np.random.randn(N_inner)
    psi = logistic(z)

    p1 = masked.psi_to_birkhoff(psi, preprocessed_mask)
    P1 = np.zeros((K, K))
    P1[rows, cols] = p1
    check_doubly_stochastic(P1)

    # plt.figure()
    # plt.subplot(111)
    # plt.imshow(P1, interpolation="nearest")
    # plt.title("Masked")
    # plt.show()

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

def test_mask():
    K = 10
    M = random_mask(K, per_row=5)
    bM = border_mask(K, per_row=5)
    dM = diagonal_mask(K, per_row=5)

    plt.figure()
    plt.subplot(131)
    plt.imshow(M, interpolation="nearest")
    plt.title("Random Mask")

    plt.subplot(132)
    plt.imshow(bM, interpolation="nearest")
    plt.title("Border Mask")

    plt.subplot(133)
    plt.imshow(dM, interpolation="nearest")
    plt.title("Diagonal Mask")

    # Count the number of possible assignments
    print("Num possible assignments: ", M.sum())
    plt.show()

if __name__ == "__main__":
    # test_mask()
    test_psi_to_birkhoff_maksed1()
    # for itr in range(100):
    #     test_psi_to_birkhoff_maksed2()