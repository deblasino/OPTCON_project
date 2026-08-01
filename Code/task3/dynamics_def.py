import sympy as sp
import numpy as np

# Define symbolic variables
ns = 8  # Number of states
ni = 2  # Number of inputs
dt = 1e-4  # Discretization step size - Forward Euler

# Define system parameters
params = {
    'm': 0.1,
    'm_act': 0.4,
    'd': 0.3,
    'alpha': 128 * 0.15,
    'c': 0.1,
    'dt': dt
}

# Define symbolic state and input variables
xx_symbolic = sp.symbols('x0:8')  # 8 state variables
uu_symbolic = sp.symbols('u0:2')# 2 input variables

# Dynamics computation
def compute_symbolic_dynamics():
    # Unpack parameters
    m = params['m']
    m_act = params['m_act']
    d = params['d']
    alpha = params['alpha']
    c = params['c']
    dt = params['dt']


    xxp_symbolic = sp.zeros(ns, 1)

    # Dynamics equations in symbolic form
    xxp_symbolic[0] = xx_symbolic[0] + dt * xx_symbolic[4]
    xxp_symbolic[1] = xx_symbolic[1] + dt * xx_symbolic[5]
    xxp_symbolic[2] = xx_symbolic[2] + dt * xx_symbolic[6]
    xxp_symbolic[3] = xx_symbolic[3] + dt * xx_symbolic[7]

    L0init= d
    L01 =d
    L02 = 2*d
    L03 = 3*d
    L0end= 4*d



    xxp_symbolic[4] = xx_symbolic[4] + dt * ((1 / m) * (
            -c * xx_symbolic[4] - alpha * (
            ((xx_symbolic[0] - 0) / (L0init * (L0init ** 2 - (xx_symbolic[0] - 0) ** 2 ))) +
            ((xx_symbolic[0] - xx_symbolic[1]) / (L01 * (L01 ** 2 - (xx_symbolic[0] - xx_symbolic[1]) ** 2 ))) +
            ((xx_symbolic[0] - xx_symbolic[2]) / (L02 * (L02 ** 2 - (xx_symbolic[0] - xx_symbolic[2]) ** 2 ))) +
            ((xx_symbolic[0] - xx_symbolic[3]) / (L03 * (L03 ** 2 - (xx_symbolic[0] - xx_symbolic[3]) ** 2 ))) +
            ((xx_symbolic[0] - 0) / (L0end * (L0end ** 2 - (xx_symbolic[0] - 0) ** 2 )))
    )
    ))
    L1init = 2*d
    L12 =d
    L13 = 2*d
    L1end = 3*d

    xxp_symbolic[5] = xx_symbolic[5] + dt * ((1 / m_act) * (
            uu_symbolic[0] - c * xx_symbolic[5] - alpha * (
            ((xx_symbolic[1] - 0) / (L1init * (L1init ** 2 - (xx_symbolic[1] - 0) ** 2 ))) +
            ((xx_symbolic[1] - xx_symbolic[0]) / (L01 * (L01 ** 2 - (xx_symbolic[1] - xx_symbolic[0]) ** 2 ))) +
            ((xx_symbolic[1] - xx_symbolic[2]) / (L12 * (L12 ** 2 - (xx_symbolic[1] - xx_symbolic[2]) ** 2 ))) +
            ((xx_symbolic[1] - xx_symbolic[3]) / (L13 * (L13 ** 2 - (xx_symbolic[1] - xx_symbolic[3]) ** 2 ))) +
            ((xx_symbolic[1] - 0) / (L1end * (L1end ** 2 - (xx_symbolic[1] - 0) ** 2 )))
    )
    ))

    L2init = 3*d
    L23 = d
    L2end = 2*d

    xxp_symbolic[6] = xx_symbolic[6] + dt *( (1 / m) * (
            -c * xx_symbolic[6] - alpha * (
            ((xx_symbolic[2] - 0) / (L2init * (L2init ** 2 - (xx_symbolic[2] - 0) ** 2 ))) +
            ((xx_symbolic[2] - xx_symbolic[0]) / (L02 * (L02 ** 2 - (xx_symbolic[2] - xx_symbolic[0]) ** 2 ))) +
            ((xx_symbolic[2] - xx_symbolic[1]) / (L12 * (L12 ** 2 - (xx_symbolic[2] - xx_symbolic[1]) ** 2 ))) +
            ((xx_symbolic[2] - xx_symbolic[3]) / (L23 * (L23 ** 2 - (xx_symbolic[2] - xx_symbolic[3]) ** 2 ))) +
            ((xx_symbolic[2] - 0) / (L2end * (L2end ** 2 - (xx_symbolic[2] - 0) ** 2 )))
    )
    ))

    L3init = 4*d
    L3end = d
    xxp_symbolic[7] = xx_symbolic[7] + dt * ((1 / m_act) * (
            uu_symbolic[1] - c * xx_symbolic[7] - alpha * (
            ((xx_symbolic[3] - 0) / (L3init * (L3init ** 2 - (xx_symbolic[3] - 0) ** 2 ))) +
            ((xx_symbolic[3] - xx_symbolic[0]) / (L03 * (L03 ** 2 - (xx_symbolic[3] - xx_symbolic[0]) ** 2 ))) +
            ((xx_symbolic[3] - xx_symbolic[1]) / (L13 * (L13 ** 2 - (xx_symbolic[3] - xx_symbolic[1]) ** 2 ))) +
            ((xx_symbolic[3] - xx_symbolic[2]) / (L23 * (L23 ** 2 - (xx_symbolic[3] - xx_symbolic[2]) ** 2 ))) +
            ((xx_symbolic[3] - 0) / (L3end * (L3end ** 2 - (xx_symbolic[3] - 0) ** 2)))
    )
    ))

    # Compute Jacobians
    jacobian_xx = xxp_symbolic.jacobian(xx_symbolic)
    jacobian_uu = xxp_symbolic.jacobian(uu_symbolic)

    return xxp_symbolic, jacobian_xx, jacobian_uu

# Precompute symbolic dynamics and lambdify
xxp_symbolic, jacobian_xx_symbolic, jacobian_uu_symbolic = compute_symbolic_dynamics()
xxp_func = sp.lambdify((xx_symbolic, uu_symbolic, list(params.keys()) ), xxp_symbolic, 'numpy')
jacobian_xx_func = sp.lambdify((xx_symbolic, uu_symbolic, list(params.keys())), jacobian_xx_symbolic, 'numpy')
jacobian_uu_func = sp.lambdify((xx_symbolic, uu_symbolic, list(params.keys()) ), jacobian_uu_symbolic, 'numpy')

# Numerical evaluation function
def dynamics(xx_vals, uu_vals):

    xxp_numeric = np.array(xxp_func(xx_vals, uu_vals, list(params.values())), dtype=float)
    jacobian_xx_numeric = np.array(jacobian_xx_func(xx_vals, uu_vals, list(params.values())), dtype=float)
    jacobian_uu_numeric = np.array(jacobian_uu_func(xx_vals, uu_vals, list(params.values()) ), dtype=float)


    return xxp_numeric, jacobian_xx_numeric, jacobian_uu_numeric

