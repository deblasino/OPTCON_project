
import numpy as np
import dynamics_def as dyn

ns = dyn.ns
ni = dyn.ni

diagonal_elements = [10000, 10000, 10000, 10000, 0.01, 0.01, 0.01, 0.01]
QQt = np.diag(diagonal_elements)
RRt = 0.01* np.eye(ni)
QQT = QQt


def stagecost(xx, uu, xx_ref, uu_ref):
    """
    Stage-cost

    Quadratic cost function
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)


    Returns:
        - ll: scalar, cost at xx, uu
        - lx: np.array, gradient of l wrt x
        - lu: np.array, gradient of l wrt u
        - lxx: np.array, Hessian of l wrt x
        - luu: np.array, Hessian of l wrt u
        - lux: np.array, mixed Hessian wrt u and x (zero for separable cost)
    """

    xx = xx[:, None]
    uu = uu[:, None]
    xx_ref = xx_ref[:, None]
    uu_ref = uu_ref[:, None]

    # Cost function
    ll = 0.5 * (xx - xx_ref).T @ QQt @ (xx - xx_ref) + 0.5 * (uu - uu_ref).T @ RRt @ (uu - uu_ref)

    # Gradients
    lx = QQt @ (xx - xx_ref)
    lu = RRt @ (uu - uu_ref)

    # Hessians
    lxx = QQt  # Hessian wrt x
    luu = RRt  # Hessian wrt u
    lux = np.zeros((ni, ns))  # Mixed Hessian is zero since the cost is separable

    return ll.squeeze(), lx.squeeze(), lu.squeeze(), lxx, luu, lux


def termcost(xx, xx_ref):
    """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Returns:
        - llT: scalar, terminal cost at xx
        - lTx: np.array, gradient of l_T wrt x
        - lTxx: np.array, Hessian of l_T wrt x
    """

    xx = xx[:, None]
    xx_ref = xx_ref[:, None]

    # Terminal cost function
    llT = 0.5 * (xx - xx_ref).T @ QQT @ (xx - xx_ref)

    # Gradient
    lTx = QQT @ (xx - xx_ref)

    # Hessian
    lTxx = QQT  # Hessian wrt x

    return llT.squeeze(), lTx.squeeze(), lTxx