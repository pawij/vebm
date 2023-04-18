# ### Simple tests
# def sanity_check():
#     """
#     Make sure everything runs
#     """
#     K = 4
#     mu = np.zeros((K - 1, K - 1))
#     sigma = 0.1 * np.ones((K - 1, K - 1))
#     Psi = mu + sigma * np.random.randn(K - 1, K - 1)
#
#     # Convert Psi to P via stick breaking raster scan
#     P = psi_to_pi_list(Psi)
#     check_doubly_stochastic_stable(P)
#     log_density_pi(P, mu, sigma)
#
#     test_log_det_jacobian(P)
#     test_plot_jacobian(P)
#
# def test_log_det_jacobian(P, tol=TOL, verbose=False):
#     """
#     Test the log det Jacobian calculation with Autograd
#     """
#     N = P.shape[0]
#     jac = jacobian(pi_to_psi_list)(P[:-1, :-1])
#     jac = jac.reshape(((N - 1) ** 2, (N - 1) ** 2))
#     sign, logdet_ag = np.linalg.slogdet(jac)
#     assert sign == 1.0
#
#     logdet_man = log_det_jacobian(P, tol=tol, verbose=verbose)
#
#     print("log det autograd: ", logdet_ag)
#     print("log det manual:   ", logdet_man)
#     assert np.allclose(logdet_ag, logdet_man, atol=1e0)
#
# def test_plot_jacobian(P):
#     jac = jacobian(pi_to_psi_list)
#     J = jac(P[:-1,:-1])
#     J = J.reshape(((K - 1) ** 2, (K - 1) ** 2))
#
#     plt.imshow(J, interpolation="none")
#     plt.xlabel("$\\mathrm{vec}(\\Pi_{1:K-1,1:K-1})$")
#     plt.ylabel("$\\mathrm{vec}(\\Psi)$")
#     plt.title("Lower triangular Jacobian")
#     plt.colorbar()
#     plt.show()
