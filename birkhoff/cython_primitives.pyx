# Cythonized primitives
#
# distutils: extra_compile_args = -O3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=False

import numpy as np
cimport numpy as np

cdef double EPS = 1e-8

cpdef cython_psi_to_pi(double[::1] psi,
                       double[::1] ubs,
                       double[::1] pi):
    """
    Convert psi to a probability vector pi
    :param psi:     Length K-1 vector in [0,1]
    :return:        Length K normalized probability vector
    """
    cdef int K = psi.size + 1
    cdef int k

    # Set pi[1..K-1]
    ubs[0] = 1.0
    for k in range(K-1):
        pi[k] = psi[k] * ubs[k]
        ubs[k+1] = ubs[k] - pi[k]

    # Set the last output
    pi[K-1] = ubs[K-1]

def cython_jacobian_psi_to_pi(double[::1] psi,
                              double[::1] ubs,
                              double[::1] pi,
                              double[:,::1] J):
    """
    J = [  -- dpi_1 / dpsi -- ]
        [  -- dpi_2 / dpsi -- ]
        [         ...         ]
        [  -- dpi_K / dpsi -- ]

    Output is K x K-1 matrix
    """
    # Initialize output
    cdef int K = psi.size + 1
    cdef int ii, k

    # Initialize intermediates
    cdef double[::1] dpsi
    cdef double[::1] dubs
    cdef double[::1] dpi


    # Run once to get intermediate computations
    cython_psi_to_pi(psi, ubs, pi)

    for ii in range(K-1):
        dpsi = np.zeros(K - 1)
        dpi = np.zeros(K)
        dubs = np.zeros(K)
        dpsi[ii] = 1.0

        # Set pi[1..K-1]
        for k in range(K - 1):
            dpi[k] = dpsi[k] * ubs[k] + psi[k] * dubs[k]
            dubs[k + 1] = dubs[k] - dpi[k]

        # Set the last output
        dpi[K - 1] = dubs[K - 1]

        # Fill in Jacobian
        for k in range(K):
            J[k, ii] = dpi[k]

cpdef cython_psi_to_birkhoff(double[:, ::1] Psi,
                             double[:, ::1] P,
                             double[:, ::1] ub_rows,
                             double[:, ::1] ub_cols,
                             double[:, ::1] lbs):
    """
    Transform a (K-1) x (K-1) matrix Psi into a KxK doubly
    stochastic matrix Pi.
    """
    cdef int K = Psi.shape[0] + 1

    # Loop indices
    cdef int i, j, k

    # Initialize
    for i in range(K):
        ub_rows[i, 0] = 1.0
        ub_cols[0, i] = 1.0

    for i in range(K - 1):
        for j in range(K - 1):
            # Compute lower bound
            lbs[i, j] = ub_rows[i, j]
            for k in range(j+1, K):
                lbs[i, j] -= ub_cols[i, k]

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
        P[i, K-1] = ub_rows[i, K-1]
        ub_cols[i+1, K-1] = ub_cols[i, K-1] - P[i, K-1]

    # Finish off the columns
    for j in range(K-1):
        P[K-1, j] = ub_cols[K-1, j]

    # Compute the bottom right entry
    # P[-1,-1] = 1 - np.sum(ub_cols[-1,:-1])
    P[K-1, K-1] = 1.0
    for k in range(K-1):
        P[K-1, K-1] -= ub_cols[K-1, k]

cpdef cython_jacobian_psi_to_birkhoff(double[:, ::1] Psi,
                                      double[:, ::1] P,
                                      double[:, ::1] ub_rows,
                                      double[:, ::1] ub_cols,
                                      double[:, ::1] lbs,
                                      double[:, :, :, ::1] J):
    """
    As above, use dual numbers to perform forward mode
    differentiation.
    """
    cdef int K = Psi.shape[0] + 1
    cdef double[:, ::1] dPsi
    cdef double[:, ::1] dP
    cdef double[:, ::1] dub_rows
    cdef double[:, ::1] dub_cols
    cdef double[:, ::1] dlbs
    cdef ii, jj, i, j, k

    # Run once to get the intermediate values
    cython_psi_to_birkhoff(Psi, P, ub_rows, ub_cols, lbs)

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
                    # dlbs[i, j] = dub_rows[i, j] - dub_cols[i,j+1:].sum()
                    dlbs[i, j] = dub_rows[i, j]
                    for k in range(j+1, K):
                        dlbs[i, j] -= dub_cols[i, k]

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
                dP[i, K-1] = dub_rows[i, K-1]
                dub_cols[i+1, K-1] = dub_cols[i, K-1] - dP[i, K-1]

            # Finish off the columns
            for j in range(K-1):
                dP[K-1, j] = dub_cols[K-1, j]

            # Compute the bottom right entry
            for k in range(K-1):
                dP[K-1,K-1] -= dub_cols[K-1,k]

            # Fill in the output
            for i in range(K):
                for j in range(K):
                    J[i,j,ii,jj] = dP[i, j]

cpdef double cython_log_det_jacobian(double[:, ::1] P,
                                     double[:, ::1] ub_rows,
                                     double[:, ::1] ub_cols,
                                     double[:, ::1] lbs):
    """
    Compute log det of the jacobian of the inverse transformation.
    Evaluate it at the given value of Pi.
    """
    cdef int K = P.shape[0]

    # Initialize output
    cdef double logdet = 0
    cdef int i, j, k

    # Initialize
    for i in range(K):
        ub_rows[i, 0] = 1.0
        ub_cols[0, i] = 1.0

    for i in range(K - 1):
        for j in range(K - 1):
            # Compute lower bound
            # lbs[i, j] = ub_rows[i, j] - ub_cols[i, j + 1:].sum()
            lbs[i, j] = ub_rows[i, j]
            for k in range(j+1, K):
                lbs[i, j] -= ub_cols[i, k]

            # Four cases
            if ub_rows[i, j] < ub_cols[i, j]:
                if lbs[i, j] > 0:
                    logdet -= np.log(ub_rows[i, j] - lbs[i, j] + EPS)
                else:
                    logdet -= np.log(ub_rows[i, j] + EPS)
            else:
                if lbs[i, j] > 0:
                    logdet -= np.log(ub_cols[i, j] - lbs[i, j] + EPS)
                else:
                    logdet -= np.log(ub_cols[i, j] + EPS)

            # Update upper bounds
            ub_rows[i, j + 1] = ub_rows[i, j] - P[i, j]
            ub_cols[i + 1, j] = ub_cols[i, j] - P[i, j]

        # Finish off the row
        ub_cols[i + 1, K-1] = ub_cols[i, K-1] - P[i, K-1]

    return logdet


def cython_grad_log_det_jacobian(double[:, ::1] P,
                                 double[:, ::1] ub_rows,
                                 double[:, ::1] ub_cols,
                                 double[:, ::1] lbs,
                                 double[:, ::1] dlogdet):
    """
    Gradient of the log det calculation.
    """
    cdef int K = P.shape[0]
    cdef int ii, jj, i, j, k

    # Initialize intermediates
    cdef double[:, ::1] dP = np.zeros((K, K))
    cdef double[:, ::1] dub_rows = np.zeros((K, K))
    cdef double[:, ::1] dub_cols = np.zeros((K, K))
    cdef double[:, ::1] dlbs = np.zeros((K, K))

    # Call once to get intermediates
    cython_log_det_jacobian(P, ub_rows, ub_cols, lbs)

    for ii in range(K):
        for jj in range(K):
            # Initialize intermediate values
            for i in range(K):
                for j in range(K):
                    dP[i, j] = 0
                    dub_rows[i, j] = 0
                    dub_cols[i, j] = 0
                    dlbs[i, j] = 0
            dP[ii, jj] = 1

            for i in range(K - 1):
                for j in range(K - 1):
                    # Compute lower bound
                    # dlbs[i, j] = dub_rows[i, j] - dub_cols[i, j + 1:].sum()
                    dlbs[i, j] = dub_rows[i, j]
                    for k in range(j+1, K):
                        dlbs[i, j] -= dub_cols[i, k]


                    # Four cases
                    if ub_rows[i, j] < ub_cols[i, j]:
                        if lbs[i, j] > 0:
                            dlogdet[ii, jj] -= (dub_rows[i, j] - dlbs[i, j]) / \
                                               (ub_rows[i, j] - lbs[i, j] + EPS)

                        else:
                            dlogdet[ii, jj] -= dub_rows[i, j] / (ub_rows[i, j] + EPS)
                    else:
                        if lbs[i, j] > 0:
                            dlogdet[ii, jj] -= (dub_cols[i, j] - dlbs[i, j]) / \
                                               (ub_cols[i, j] - lbs[i, j] + EPS)
                        else:
                            dlogdet[ii, jj] -= dub_cols[i, j] / (ub_cols[i, j] + EPS)

                    # Update upper bounds
                    dub_rows[i, j + 1] = dub_rows[i, j] - dP[i, j]
                    dub_cols[i + 1, j] = dub_cols[i, j] - dP[i, j]

                # Finish off the row
                dub_cols[i + 1, K-1] = dub_cols[i, K-1] - dP[i, K-1]
