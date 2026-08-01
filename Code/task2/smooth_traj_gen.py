import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from equilibrium_comp import find_equilibrium
def generate_reference_trajectory(tf=1, dt=1e-4):

    # Parameters
    TT = int(tf / dt)
    ns = 4  # Number of positions (z0, z1, z2, z3)
    ni = 2

    positions = [
        [0, 0],
        [0, 0],
        [0.002,0.002],
        [0.005,0.005],
        [0.002, 0.002],
        [0,0],
        [0, 0]
    ]

    #list for equilibrium states and inputs
    xx = []  # Stati: [z0, z1, z2, z3]
    uu = []  # Input: [u0, u1]

    # compute equilibria
    for pos in positions:
        z1,z3 = pos
        try:
            [z0,z1,z2,z3],[u0,u1] = find_equilibrium( z1, z3)
            xx.append([ z0,z1,z2, z3])
            uu.append([u0, u1])
        except RuntimeError as e:
            print(e)
            continue

    xx = np.array(xx)
    uu = np.array(uu)

    # time
    t_equilibria = np.linspace(0, tf, len(xx))

    # time for interpolation
    t_interp = np.arange(0, tf, dt)

    # Interpolation
    xxref = np.zeros((ns, len(t_interp)))
    for i in range(ns): #interpolate each state
        cs = CubicSpline(t_equilibria, xx[:, i])
        xxref[i, :] = cs(t_interp)


    uuref = np.zeros((ni, len(t_interp)))
    for i in range(ni):
        cs = CubicSpline(t_equilibria, uu[:, i])
        uuref[i, :] = cs(t_interp)

    # add four rows of zeros, representing the velocities = 0
    velocities = np.zeros_like(xxref)
    xxref = np.vstack([xxref, velocities])

    return xxref, uuref


""" 
# TO PLOT REFERENCE TRAJECTORY
if __name__ == "__main__":
    
    xxref, uuref, t_interp, xx, uu, t_equilibria = generate_reference_trajectory()

    
    print("dimension of xxref:", xxref.shape)
    print("dimension of  uuref:", uuref.shape)

    # Plot
    plt.figure(figsize=(12, 8))
    for i in range(4):  
        plt.subplot(2, 2, i + 1)
        plt.plot(t_interp, xxref[i, :], label=f"z{i-1} interpolated", color="blue")
        plt.scatter(t_equilibria, xx[:, i], color="red", label=f"z{i} equilibrium")
        plt.xlabel("Time (s)")
        plt.ylabel(f"z{i}")
        plt.legend()
        plt.grid(True)
    plt.suptitle("Interpolated desired positions")
    plt.tight_layout()

    # Plot 
    plt.figure(figsize=(12, 4))
    for i in range(2):  
        plt.subplot(1, 2, i + 1)
        plt.plot(t_interp, uuref[i, :], label=f"u{i} interpolated", color="blue")
        plt.scatter(t_equilibria, uu[:, i], color="red", label=f"u{i} equilibrium")
        plt.xlabel("Time (s)")
        plt.ylabel(f"u{i}")
        plt.legend()
        plt.grid(True)
    plt.suptitle("interpolated input")
    plt.tight_layout()

    plt.show()
    
    """