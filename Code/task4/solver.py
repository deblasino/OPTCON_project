
import numpy as np
import cvxpy as cp
import time


def solver_linear_mpc(AA, BB, QQ, RR, QQf, xxt, x_ref, u_ref, umax = 1, umin = -1,  T_pred = 5):

    xxt = xxt.squeeze()

    ns, ni = BB.shape

    xx_mpc = cp.Variable((ns, T_pred))
    uu_mpc = cp.Variable((ni, T_pred))

    cost = 0
    constr = []

    for tt in range(T_pred-1):

        cost += cp.quad_form(xx_mpc[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt], RR)
        constr += [xx_mpc[:,tt+1] == AA@xx_mpc[:,tt] + BB@uu_mpc[:,tt], # dynamics constraint
                uu_mpc[:,tt]+u_ref[:,tt] <= umax, # input constraints
                uu_mpc[:,tt]+u_ref[:,tt] >= umin,
                    ]
    # sums problem objectives and concatenates constraints.
    cost += cp.quad_form(xx_mpc[:,T_pred-1], QQf)

    constr += [xx_mpc[:, 0] == xxt - x_ref[:, 0]]

    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve( solver=cp.OSQP,warm_start =False)

    if problem.status == "infeasible":
        print("Infeasible problem. CHECK THE CONSTRAINTS.")


    return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value