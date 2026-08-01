import numpy as np
import dynamics_def as dyn
import equilibrium_comp as eq
import matplotlib.pyplot as plt

# Define system parameters
dt = dyn.dt
ns = dyn.ns
ni = dyn.ni
params = {
    'm': 0.1,
    'm_act': 0.4,
    'd': 0.3,
    'alpha': 128 * 0.15,
    'c': 0.1,
    'dt': dt
}

def gen(tf, dt, ns, ni):

    TT = int(tf / dt)

    equilibrium_positions, equilibrium_inputs = eq.find_equilibrium(z1=0.003, z3=0.00225)
    target_eq = np.append(equilibrium_positions, np.zeros(4))


    target_eq_u = np.array(equilibrium_inputs).reshape(ni, 1)  # Reshape to (2, 1)
    print(f"Equilibrium input is :{equilibrium_inputs}")

    #####################
    # MAKE A REFERENCE TRAJECTORY BETWEEN THE POINT
    #####################

    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))  # it remains zero cause in both equilibria u is zero

    # Midpoint index of the trajectory
    mid_point = int(TT / 2)

    # Assign xxT to the reference trajectory from the middle point onwards
    xx_ref[0:ns, mid_point:] = np.tile(target_eq.reshape(ns, 1), (1, TT - mid_point))
    uu_ref[0:ni, mid_point:] = np.tile(target_eq_u, (1, TT - mid_point))  # Corrected shape

    return xx_ref, uu_ref


#PLOT DESIRED STATES WHEN CALLED
"""

xx_ref, uu_ref = gen(tf=1, dt=dt, ns=ns, ni=ni)

# Create time steps for plotting (x-axis)
time_steps = np.arange(xx_ref.shape[1]) * dt

# Plot each state separately in individual graphs
for i in range(xx_ref.shape[0] - 4):  # Assuming you want to exclude the last 4 states
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, xx_ref[i, :], label=f"State {i + 1}", color='blue')
    plt.xlabel("Time [s]")
    plt.ylabel("Value")
    plt.title(f"Reference Trajectory for State {i + 1}")
    plt.legend()
    plt.grid()
    plt.show()

"""