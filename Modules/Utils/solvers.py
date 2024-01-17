import numpy as np
from scipy import sparse
from scipy.linalg import norm, solve_triangular
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.misc import LinAlgError

def Thomas(A, b):
    """
    Solver for the linear equation Ax=b using the Thomas algorithm.
    It is a wrapper of the LAPACK function dgtsv.
    """

    D = A.diagonal(0)
    L = A.diagonal(-1)
    U = A.diagonal(1)

    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("expected square matrix")
    if A.shape[0] != b.shape[0]:
        raise ValueError("incompatible dimensions")
    
    (dgtsv,) = get_lapack_funcs(("gtsv",))
    du2, d, du, x, info = dgtsv(L, D, U, b)

    if info == 0:
        return x
    if info > 0:
        raise LinAlgError("singular matrix: resolution failed at diagonal %d" % (info - 1))