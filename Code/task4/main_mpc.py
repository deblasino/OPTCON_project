import numpy as np
import matplotlib.pyplot as plt
import dynamics_def as dyn
from solver import solver_linear_mpc

ns = dyn.ns
ni = dyn.ni
dt= dyn.dt
tf = 1

x_ref = np.load('xx_star_1e-4_1s_def.npy')
u_ref = np.load('uu_star_1e-4_1s_def.npy')

###############################
# MPC Parameters Initialization
###############################

qp_solve_freq = 10

#assigne a perturbed initial condition
xx0 = np.array([-0.002, -0.002, -0.002, -0.002, 0, 0, 0, 0])
#xx0 = np.zeros((ns,1))

Tsim = int(tf/dt) # simulation horizon, make equal to samples of traj to plot whole trajectory

T_pred = 5   # MPC Prediction horizon
umax =5  # Upper input constraint0
umin = -umax  # Lower input constraint


time = np.linspace(0, tf, Tsim)




########################
# Cost
#######################

diagonal_elements = [1000, 1000, 1000, 1000, 500, 500, 500, 500]
Q_t = np.diag(diagonal_elements)
R_t = 0.01 * np.eye(ni)
Q_T = Q_t



# extend the reference trajectory if necessary
if x_ref.shape[1] < Tsim:
    x_ref_extended = np.zeros((ns, Tsim))
    u_ref_extended = np.zeros((ni, Tsim))
    for i in range(ns):
        x_ref_extended[i, :] = np.interp(np.arange(Tsim), np.linspace(0, Tsim, x_ref.shape[1]), x_ref[i, :])
    for i in range(ni):
        u_ref_extended[i, :] = np.interp(np.arange(Tsim), np.linspace(0, Tsim, u_ref.shape[1]), u_ref[i, :])
    x_ref = x_ref_extended
    u_ref = u_ref_extended

# truncate if x_ref and u_ref are longer than Tsim
if x_ref.shape[1] > Tsim:
    x_ref = x_ref[:, :Tsim]
    u_ref = u_ref[:, :Tsim]

#############################
# Model Predictive Control
#############################

xx_real_mpc = np.zeros((ns, Tsim))
uu_real_mpc = np.zeros((ni, Tsim))
err = np.zeros((ns, Tsim))

xx_mpc = np.zeros((ns, T_pred, Tsim))

xx_real_mpc[:, 0] = xx0.squeeze()
print(f"Initial condition is {xx_real_mpc[:,0]}")


for tt in range(Tsim - 1):
    # system evolution  real with MPC

    # ensure the slicing does not exceed the length of x_ref and u_ref
    max_slice = min(T_pred, Tsim - tt)  # Adjust the slice size if near the end

    x_ref_pred = x_ref[:, tt:tt + max_slice]
    u_ref_pred = u_ref[:, tt:tt + max_slice]

    # pad the prediction window if it's shorter than T_pred
    if x_ref_pred.shape[1] < T_pred:
        x_ref_pred = np.hstack((x_ref_pred, np.tile(x_ref_pred[:, -1:], (1, T_pred - x_ref_pred.shape[1]))))
        u_ref_pred = np.hstack((u_ref_pred, np.tile(u_ref_pred[:, -1:], (1, T_pred - u_ref_pred.shape[1]))))


    dfx, dfu = dyn.dynamics(xx_real_mpc[:, tt], uu_real_mpc[:, tt-1])[1:3]

    A_t = dfx
    B_t = dfu

    xx_t_mpc = xx_real_mpc[:, tt]  # get initial condition

    # solve MPC problem apply first input
    if tt % 100 == 0:  # print every 100 time instants
        print('MPC:\t t = {}'.format(tt))

    if tt % qp_solve_freq == 0:  #compute control every 10 iterations
        uu_real_mpc[:, tt], xx_mpc[:, :, tt] = solver_linear_mpc(A_t, B_t, Q_t, R_t, Q_T, xx_real_mpc[:, tt], x_ref_pred,
                                                                 u_ref_pred,
                                                                 umax=umax, umin=umin,
                                                                  T_pred=T_pred)[:2]

    else:
        uu_real_mpc[:, tt] = uu_real_mpc[:, tt - 1]

    xx_real_mpc[:, tt + 1] = dyn.dynamics(xx_real_mpc[:, tt], u_ref[:,tt]+uu_real_mpc[:, tt])[0].squeeze()

    err[:, tt] = xx_real_mpc[:, tt] - x_ref[:, tt]



###############
#ERROR ANALYSIS
###############

time = np.linspace(0, tf, Tsim)


print('Maximum position error on second half:', np.max(abs(err[:4, int((Tsim-1)/2):])))
print('Maximum velocity error on second half:', np.max(abs(err[4:, int((Tsim-1)/2):])))

print('Mean position error:', np.mean(abs(err[:4, :])))
print('Mean velocity error:', np.mean(abs(err[4:, :])))

print('Mean position error on first half:', np.mean(abs(err[:4,0:int((Tsim-1)/2)])))
print('Mean velocity error on first half:', np.mean(abs(err[4:, 0:int((Tsim-1)/2)])))


#CONVERGENCE TIME
# Parameters
threshold_percentage = 0.1  # 10%
consecutive_steps = 2000    # Consecutive steps below threshold, 20%of the total duration of trajectory ( 0.2 seconds in this case) Given

# Maximum error for each position (on first 100 positions)
window_to_check = 100
max_initial_error_x0 = np.max(np.abs(err[:1, :window_to_check]))
convergence_threshold_x0 = threshold_percentage * max_initial_error_x0
max_initial_error_x1 = np.max(np.abs(err[1:2, :window_to_check]))
convergence_threshold_x1 = threshold_percentage * max_initial_error_x1
max_initial_error_x2 = np.max(np.abs(err[2:3, :window_to_check]))
convergence_threshold_x2 = threshold_percentage * max_initial_error_x2
max_initial_error_x3 = np.max(np.abs(err[3:4, :window_to_check]))
convergence_threshold_x3 = threshold_percentage * max_initial_error_x3

# Variables
convergence_time = None
counter = 0

# Find convergence
for tt in range(Tsim):
    current_error_x0 = abs(err[0, tt])
    current_error_x1 = abs(err[1, tt])
    current_error_x2 = abs(err[2, tt])
    current_error_x3 = abs(err[3, tt])

    if current_error_x0 < convergence_threshold_x0 and current_error_x1 < convergence_threshold_x1 and current_error_x2 < convergence_threshold_x2 and current_error_x3 < convergence_threshold_x3:
        counter += 1
        if counter >= consecutive_steps:
            convergence_time = time[tt-consecutive_steps]  # Initial convergence time
            break
    else:
        counter = 0  # Reset if error too high

# Print
if convergence_time is not None:
    print(
        f"\nConvergence time: {convergence_time:.2f} s (threshold: {threshold_percentage * 100}% of maximum initial error)")
else:
    print("\nNo convergence in this time interval.")




#######################################
# Plots
#######################################




#plot tracking positions
plt.figure(figsize=(10, 8))
plt.subplots_adjust(hspace=0.5)
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(time, xx_real_mpc[i, :], label=f'z{i+1} MPC', color='blue')
    plt.plot(time, x_ref[i, :], '--r', label=f'z{i+1} Ref', linewidth=2)
    plt.xlabel('Time [s]', fontsize=8)
    plt.ylabel(f'z{i+1}', fontsize=8)
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
plt.suptitle('Positions', fontsize=12)
plt.show()

# plot tracking velocities
plt.figure(figsize=(10, 8))
plt.subplots_adjust(hspace=0.5)
for i in range(4, 8):
    plt.subplot(4, 1, i - 3)
    plt.plot(time, xx_real_mpc[i, :], label=f'$\dot{{z}}_{i+1-4}$ MPC', color='blue')
    plt.plot(time, x_ref[i, :], '--r', label=f'$\dot{{z}}_{i+1-4}$ Ref', linewidth=2)
    plt.xlabel('Time [s]', fontsize=8)
    plt.ylabel(f'$\dot{{z}}_{i+1-4}$', fontsize=12)
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
plt.suptitle('Velocities', fontsize=12)
plt.show()

#plot tracking input u
plt.figure(figsize=(10, 6))
plt.subplots_adjust(hspace=0.5)
for i in range(ni):
    plt.subplot(ni, 1, i + 1)
    plt.plot(time, uu_real_mpc[i, :]+u_ref[i, :], label=f'u{i+1} MPC', color='blue')
    plt.plot(time, u_ref[i, :], '--r', label=f'u{i+1} Ref', linewidth=2)
    plt.plot(time, np.ones(Tsim) * umax, '--g', linewidth=1.5, label='Max Input')
    plt.plot(time, np.ones(Tsim) * umin, '--g', linewidth=1.5, label='Min Input')
    plt.xlabel('Time [s]', fontsize=8)
    plt.ylabel(f'u{i+1}', fontsize=8)
    plt.legend(loc='upper right', fontsize='small')
    plt.grid(True)
plt.suptitle('Input', fontsize=12)
plt.show()

plt.figure(figsize=(8,6))

# Plot tracking errors
plt.figure(figsize=(8, 6))
plt.subplots_adjust(hspace=0.5)
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(time, err[i, :], label=f'z{i+1}', color='blue')
    if i == 0:
        plt.title(f'Tracking Error', fontsize=10)
    plt.xlabel('Time [s]', fontsize=8)
    if i ==1:
        plt.ylabel('Tracking Error')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.grid(True)
plt.show()