"""
Forward mode differentiation for stick breaking transformations.
In the stick breaking transformations, all of the computations are
addition and subtraction of scalars.  This makes it pretty easy to
do the forward mode autodiff
"""
import autograd.numpy as np
from autograd.core import primitive, Node

import birkhoff.cython_primitives as cython_primitives

NOT_DETERMINED = 0
ROW_DETERMINED = 1
COL_DETERMINED = 2
BOTH_DETERMINED = 3

### Preprocess a mask

def order_from_mask(M):
    K = M.shape[0]
    assert M.shape == (K, K)
    assert M.dtype == bool

    # Simple sanity check
    # (necessary but not sufficient
    # conditions for satisfiability)
    assert np.all(M.sum(0) >= 1)
    assert np.all(M.sum(1) >= 1)

    # Output order = [(i1, j1), ..., (in, jn)]
    # i.e. the ordered list of entries to fill in
    # If M[i,j] = 0, do *not* include (i,j) in order.
    # order only includes the nonzero entries of M.
    rows = []
    cols = []

    # Output whether or not this entry is predetermined
    # by what has been filled in before.  This will be
    # a list of indicators for (not determined, determined by row,
    # or determined by column).
    determined = []

    # Keep track of which entries we've processed
    # To start, assume we've processed the entries where
    # the mask is False
    processed = ~M

    def check_determined(i, j):
        if np.all(processed[i, :j]) and np.all(processed[i, j + 1:]):
            return ROW_DETERMINED
        elif np.all(processed[:i, j]) and np.all(processed[i + 1:, j]):
            return COL_DETERMINED
        else:
            return NOT_DETERMINED

    def resolve_children(i, j):
        # Look for entries that are now fully determined
        if processed[i, j]:
            # Base case: child already processed
            return

        d = check_determined(i, j)
        if d == NOT_DETERMINED:
            # Base case: child not fully determined
            return

        # Otherwise, child is unprocessed and fully determined
        # Append (i,j) to the list
        rows.append(i)
        cols.append(j)
        determined.append(d)
        processed[i, j] = True

        # Recurse on other entries in this row
        for k in range(K):
            if M[i, k] == True:
                resolve_children(i, k)

        # Recurse on other entries in this column
        for k in range(K):
            if M[k, j] == True:
                resolve_children(k, j)

    # Now iterate over the entries and construct the order
    for i in range(K):
        for j in range(K):
            if M[i, j] == False:
                continue

            # Check if this entry has been processed already.
            if processed[i, j]:
                continue


            # Not deterministic, add it to the list
            rows.append(i)
            cols.append(j)
            determined.append(check_determined(i, j))
            processed[i, j] = True

            # resolve its children
            for k in range(K):
                if M[i, k] == True:
                    resolve_children(i, k)

            # Recurse on other entries in this column
            for k in range(K):
                if M[k, j] == True:
                    resolve_children(k, j)

    # Convert to arrays
    rows = np.array(rows)
    cols = np.array(cols)
    determined = np.array(determined)

    assert np.all(processed)
    assert len(rows) == M.sum()
    assert len(cols) == M.sum()

    # Compute the number of undetermined values
    N = np.sum(determined == NOT_DETERMINED)
    return N, rows, cols, determined


### Stick breaking transformations for permutations
def python_psi_to_birkhoff_masked(K, psi, rows, cols, determined, verbose=True):
    """
    Transform a vector Psi into a KxK doubly
    stochastic matrix Pi.
    """
    # Get the number of possible assignments
    N = rows.size
    assert rows.shape == (N,)
    assert cols.shape == (N,)
    assert determined.shape == (N,)
    assert psi.shape == (np.sum(determined == NOT_DETERMINED), )

    # Initialize intermediate values
    # NOTE: Important that upper bounds are initialized to one!
    P = np.zeros((N,), dtype=float)
    ub_rows = np.ones((K,))
    ub_cols = np.ones((K,))

    # Loop over output values.  Keep track of pointer into psi.
    ni = 0
    for n in range(N):
        i, j = rows[n], cols[n]
        lb = 0

        # Check if determined by the rows or the columns
        if determined[n] == BOTH_DETERMINED:
            P[n] = min(ub_rows[i], ub_cols[j])
        elif determined[n] == ROW_DETERMINED:
            P[n] = min(ub_rows[i], ub_cols[j])
        elif determined[n] == COL_DETERMINED:
            P[n] = min(ub_rows[i], ub_cols[j])

        else:
            # Otherwise, this is an interior point and we need to map psi -> pi.
            # Compute lower bound.  Column upper bounds are subtracted only if the
            # mask allows those columns to have entries in the i-th row.
            lb = ub_rows[i]
            for nnext in range(n+1, N):
                if rows[nnext] == i:
                    lb -= ub_cols[cols[nnext]]

            # Do we also have a lower bound from the remainder of the column?
            assert lb <= min(ub_cols[j], ub_rows[i])

            # Four cases
            if ub_rows[i] < ub_cols[j]:
                if lb > 0:
                    P[n] = lb + (ub_rows[i] - lb) * psi[ni]
                else:
                    P[n] = ub_rows[i] * psi[ni]
            else:
                if lb > 0:
                    P[n] = lb + (ub_cols[j] - lb) * psi[ni]
                else:
                    P[n] = ub_cols[j] * psi[ni]

            ni += 1

        # DEBUG
        if verbose:
            print("({},{}): det: {}   lb: {:.3f}   ubr: {:.3f}   ubc: {:.3f}  P: {:.3f}"
                  .format(i, j, determined[n], lb, ub_rows[i], ub_cols[j], P[n]))

        # Update upper bounds
        ub_rows[i] = ub_rows[i] - P[n]
        ub_cols[j] = ub_cols[j] - P[n]

        # DEBUG
        # assert ub_cols[j] >= -1e-8
        # assert ub_rows[i] >= -1e-8
        #
        # assert P[n] >= -1e-8
        # assert P[n] <= 1 + 1e-8

    # Make sure we used all the psi's
    assert ni == len(psi)

    return P

### Stick breaking transformations for permutations
def python_psi_to_birkhoff_naive(Psi, M, verbose=True):
    """
    Transform a vector Psi into a KxK doubly
    stochastic matrix Pi.
    """
    K = M.shape[0]
    assert M.shape == (K, K)
    assert M.dtype == bool

    assert Psi.shape == (K, K)
    assert np.all(Psi) >= 0
    assert np.all(Psi) <= 1

    # Initialize intermediate values
    # NOTE: Important that upper bounds are initialized to one!
    P = np.zeros((K,K), dtype=float)
    ubs = np.ones((K, K))
    lbs = np.zeros((K, K))
    processed = np.zeros((K, K), dtype=bool)

    def _resolve_bounds(i, j):
        # when(i,j) is updated, it affects the bounds of the elements
        # in the i-th row and j-th column. when these bounds are updated
        # they propagate to other rows and columns

        # nonnegativity:  P[i,j] >= 0   for all i,j in 1...K
        # row sums: \sum_k P[i,k] = 1   for all i in 1...K
        # col sums: \sum_k P[k,j] = 1   for all j in 1...K


        pass

    # Loop over entries in P. Assume that the bounds start in a valid
    # configuration, and whenever a value is set, the bounds are updated.
    for i in range(K):
        for j in range(K):
            P[i, j] = lbs[i, j] + (ubs[i,j] - lbs[i, j]) * Psi[i,j]
            _resolve_bounds(i, j)

    return P

# def cython_psi_to_birkhoff_masked(Psi, M):
#     # TODO
#     Psi = Psi.value if isinstance(Psi, Node) else Psi
#     K = Psi.shape[0] + 1
#     P = np.zeros((K, K))
#     ub_rows = np.ones((K, K))
#     ub_cols = np.ones((K, K))
#     lbs = np.zeros((K, K))
#     cython_primitives.cython_psi_to_birkhoff_masked(Psi, P, ub_rows, ub_cols, lbs)
#     return P

def python_jacobian_psi_to_birkhoff_masked(psi, preprocessed_mask):
    """
    As above, use dual numbers to perform forward mode
    differentiation.
    """
    K, N, N_inner, rows, cols, inner, rowend, colend = preprocessed_mask

    # Get the number of possible assignments
    assert psi.shape == (N_inner,)

    # Initialize the Jacobian
    J = np.zeros((N, N_inner))

    # Initialize intermediate values and derivatives
    for nni in range(N_inner):
        dpsi = np.zeros(N_inner)
        dpsi[nni] = 1.0

        P = np.zeros((N,))
        ub_row = np.ones((N,))
        ub_cols = np.ones((K,))

        # Initialize forward derivatives
        dP = np.zeros((N,))
        dub_row = np.zeros((N,))
        dub_cols = np.zeros((K,))

        # Loop over output values.  Keep track of pointer into psi.
        ni = 0
        for n in range(N):
            i, j = rows[n], cols[n]

            if rowend[n]:
                #  If this is the end of the row, its entry is determined by ub_row
                P[n] = ub_row[n]
                dP[n] = dub_row[n]

                ub_cols[j] -= P[n]
                dub_cols[j] -= dP[n]

            elif colend[n]:
                # Likewise, the end of the column is determined by ub_col
                P[n] = ub_cols[j]
                dP[n] = dub_cols[j]

                ub_cols[j] -= P[n]
                dub_cols[j] -= dP[n]

            else:
                # Otherwise, this is an interior point and we need to map psi -> pi.
                # Compute lower bound.  Column upper bounds are subtracted only if the
                # mask allows those columns to have entries in the i-th row.
                lb = ub_row[n]
                dlb = dub_row[n]

                for nnext in range(n + 1, N):
                    if rows[nnext] != i:
                        break
                    else:
                        lb -= ub_cols[cols[nnext]]
                        dlb -= dub_cols[cols[nnext]]

                # Four cases
                if ub_row[n] < ub_cols[j]:
                    if lb > 0:
                        P[n] = lb + (ub_row[n] - lb) * psi[ni]
                        dP[n] = dlb + (ub_row[n] - lb) * dpsi[ni] + (dub_row[n] - dlb) * psi[ni]
                    else:
                        P[n] = ub_row[n] * psi[ni]
                        dP[n] = dub_row[n] * psi[ni] + ub_row[n] * dpsi[ni]
                else:
                    if lb > 0:
                        P[n] = lb + (ub_cols[j] - lb) * psi[ni]
                        dP[n] = dlb + (ub_cols[j] - lb) * dpsi[ni] + (dub_cols[j] - dlb) * psi[ni]
                    else:
                        P[n] = ub_cols[j] * psi[ni]
                        dP[n] = dub_cols[j] * psi[ni] + ub_cols[j] * dpsi[ni]

                ni += 1

                # Update upper bounds
                ub_row[n + 1] = ub_row[n] - P[n]
                dub_row[n + 1] = dub_row[n] - dP[n]

                ub_cols[j] = ub_cols[j] - P[n]
                dub_cols[j] = dub_cols[j] - dP[n]

        # Store jacobian
        J[:,nni] = dP

    return J

# def cython_jacobian_psi_to_birkhoff_masked(Psi):
#     Psi = Psi.value if isinstance(Psi, Node) else Psi
#     K = Psi.shape[0] + 1
#     P = np.zeros((K, K))
#     ub_rows = np.zeros((K, K))
#     ub_cols = np.zeros((K, K))
#     lbs = np.zeros((K, K))
#     J = np.zeros((K, K, K-1, K-1))
#     cython_primitives.cython_jacobian_psi_to_birkhoff_masked(Psi, P, ub_rows, ub_cols, lbs, J)
#     return J
#
psi_to_birkhoff = primitive(python_psi_to_birkhoff_masked)
psi_to_birkhoff.defvjp(lambda g, ans, vs, gvs, Psi, preprocessed_mask, **kwargs:
                       np.dot(g, python_jacobian_psi_to_birkhoff_masked(Psi, preprocessed_mask)))

### Log determinants
def python_log_det_jacobian_masked(P, M, return_intermediates=False):
    """
    Compute log det of the jacobian of the inverse transformation.
    Evaluate it at the given value of Pi.
    :param Pi: Doubly stochastic matrix
    :return: |dPsi / dPi |
    """
    K = M.shape[0]
    assert M.shape == (K, K)
    assert M.dtype == bool
    assert np.all(M.sum(0) >= 1)
    assert np.all(M.sum(1) >= 1)

    # Initialize output
    logdet = 0

    # Initialize intermediate values
    ub_rows = np.ones((K, K))
    ub_cols = np.ones((K, K))
    lbs = np.zeros((K, K))
    for i in range(K - 1):
        for j in range(K - 1):
            if not M[i, j]:
                continue

            # Compute lower bound
            lbs[i, j] = ub_rows[i, j] - (M[i, j+1:] * ub_cols[i, j + 1:]).sum()

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

def cython_log_det_jacobian_masked(P, M):
    P = P.value if isinstance(P, Node) else P
    K = P.shape[0]
    ub_rows = np.zeros((K, K))
    ub_cols = np.zeros((K, K))
    lbs = np.zeros((K, K))
    return cython_primitives.cython_log_det_jacobian(P, ub_rows, ub_cols, lbs)

def python_grad_log_det_jacobian(P, M):
    """
    Gradient of the log det calculation.
    """
    K = P.shape[0]
    assert P.shape == (K, K)

    assert M.shape == (K, K)
    assert M.dtype == bool
    assert np.all(M.sum(0) >= 1)
    assert np.all(M.sum(1) >= 1)

    # Initialize output
    dlogdet = np.zeros((K, K))

    # Call once to get intermediates
    _, ub_rows, ub_cols, lbs = python_log_det_jacobian_masked(P, M, return_intermediates=True)

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
                    if not M[i,j] == 0:
                        continue

                    # Compute lower bound
                    dlbs[i, j] = dub_rows[i, j] - (M[i,j+1:] * dub_cols[i, j + 1:]).sum()

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

# def cython_grad_log_det_jacobian(P):
#     P = P.value if isinstance(P, Node) else P
#     K = P.shape[0]
#     ub_rows = np.zeros((K, K))
#     ub_cols = np.zeros((K, K))
#     lbs = np.zeros((K, K))
#     dlogdet = np.zeros((K, K))
#     cython_primitives.cython_grad_log_det_jacobian(P, ub_rows, ub_cols, lbs, dlogdet)
#     return dlogdet

# log_det_jacobian = primitive(cython_log_det_jacobian)
# log_det_jacobian.defvjp(lambda g, ans, vs, gvs, P, **kwargs:
#                         np.full(P.shape, g) * cython_grad_log_det_jacobian(P))

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

