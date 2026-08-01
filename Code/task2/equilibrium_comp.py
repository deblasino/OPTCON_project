import numpy as np
from scipy.optimize import root

def find_equilibrium(z1, z3):
    """
    Find equilibrium positions z0, z2 and inputs u0, u1 for given actuated positions z1 and z3.
    Returns two vectors: equilibrium positions [z0, z1, z2, z3] and equilibrium inputs [u0, u1].
    """
    # System parameters
    m = 0.1
    m_act = 0.4
    d = 0.3
    alpha = 128 * 0.15
    c = 0.1
    z_init = 0.0  # Fixed point at the start
    z_end = 0.0   # Fixed point at the end

    # Define the equilibrium equations
    def equilibrium_equations(vars):
        z0, z2, u0, u1 = vars

        # Compute distances L_ij
        L0init = d
        L01 = d
        L02 = 2 * d
        L03 = 3 * d
        L0end = 4 * d

        L1init = 2 * d
        L12 = d
        L13 = 2 * d
        L1end = 3 * d

        L2init = 3 * d
        L23 = d
        L2end = 2 * d

        L3init = 4 * d
        L3end = d

        # Equilibrium equations
        eq0 = -alpha * (
                ((z0 - z_init) / (L0init * (L0init ** 2 - (z0 - z_init) ** 2))) +
                ((z0 - z1) / (L01 * (L01 ** 2 - (z0 - z1) ** 2))) +
                ((z0 - z2) / (L02 * (L02 ** 2 - (z0 - z2) ** 2))) +
                ((z0 - z3) / (L03 * (L03 ** 2 - (z0 - z3) ** 2))) +
                ((z0 - z_end) / (L0end * (L0end ** 2 - (z0 - z_end) ** 2)))
        )

        eq1 = u0 - alpha * (
                ((z1 - z_init) / (L1init * (L1init ** 2 - (z1 - z_init) ** 2))) +
                ((z1 - z0) / (L01 * (L01 ** 2 - (z1 - z0) ** 2))) +
                ((z1 - z2) / (L12 * (L12 ** 2 - (z1 - z2) ** 2))) +
                ((z1 - z3) / (L13 * (L13 ** 2 - (z1 - z3) ** 2))) +
                ((z1 - z_end) / (L1end * (L1end ** 2 - (z1 - z_end) ** 2)))
        )

        eq2 = -alpha * (
                ((z2 - z_init) / (L2init * (L2init ** 2 - (z2 - z_init) ** 2))) +
                ((z2 - z0) / (L02 * (L02 ** 2 - (z2 - z0) ** 2))) +
                ((z2 - z1) / (L12 * (L12 ** 2 - (z2 - z1) ** 2))) +
                ((z2 - z3) / (L23 * (L23 ** 2 - (z2 - z3) ** 2))) +
                ((z2 - z_end) / (L2end * (L2end ** 2 - (z2 - z_end) ** 2)))
        )

        eq3 = u1 - alpha * (
                ((z3 - z_init) / (L3init * (L3init ** 2 - (z3 - z_init) ** 2))) +
                ((z3 - z0) / (L03 * (L03 ** 2 - (z3 - z0) ** 2))) +
                ((z3 - z1) / (L13 * (L13 ** 2 - (z3 - z1) ** 2))) +
                ((z3 - z2) / (L23 * (L23 ** 2 - (z3 - z2) ** 2))) +
                ((z3 - z_end) / (L3end * (L3end ** 2 - (z3 - z_end) ** 2)))
        )

        return [eq0, eq1, eq2, eq3]

    # Initial guess for z0, z2, u0, u1
    initial_guess = [0.0, 0.0, 0.0, 0.0]

    # Solve using scipy.optimize.root
    result = root(equilibrium_equations, initial_guess, method='lm')  # Levenberg-Marquardt method

    if result.success:
        z0, z2, u0, u1 = result.x
        equilibrium_positions = [z0, z1, z2, z3]
        equilibrium_inputs = [u0, u1]
        return equilibrium_positions, equilibrium_inputs
    else:
        raise RuntimeError("No equilibrium found for z1 = {}, z3 = {}".format(z1, z3))

# Example
if __name__ == "__main__":
    # Desired actuated positions
    z1 = 0.005
    z3 = 0.005

    try:
        equilibrium_positions, equilibrium_inputs = find_equilibrium(z1, z3)
        print("Equilibrium positions: z0 = {}, z1 = {}, z2 = {}, z3 = {}".format(*equilibrium_positions))
        print("Equilibrium inputs: u0 = {}, u1 = {}".format(*equilibrium_inputs))
    except RuntimeError as e:
        print(e)