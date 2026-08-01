import dynamics_def as dyn
import numpy as np
import matplotlib.pyplot as plt


# Number of variables of the system
ns = dyn.ns
ni = dyn.ni

tf = 1  # final time in seconds
dt = dyn.dt  # get discretization step from dynamics
TT = int(tf / dt)

#load the reference trajectories from previous task
xx_ref = np.load('xx_star_1e-4_1s_def.npy')
uu_ref = np.load('uu_star_1e-4_1s_def.npy')


# Costs for LQR problem
diagonal_elements = [1000, 1000, 1000, 1000, 0.1, 0.1, 0.1, 0.1]
Q_t = np.diag(diagonal_elements)
R_t = 0.001 * np.eye(ni)
Q_T = Q_t

P_next = Q_T

# Linearization matrices, number of matrices equal to number of point trajectories
A_t = np.zeros((ns, ns, TT))
B_t = np.zeros((ns, ni, TT))

K_t = []


for tt in reversed(range(TT - 1)):
    #  (Jacobians) of dynamics
    dfx, dfu = dyn.dynamics(xx_ref[:, tt], uu_ref[:, tt])[1:3]

    # Linearization of the system on the trajectory
    A_t[:, :, tt] = dfx
    B_t[:, :, tt] = dfu

    M_t =( R_t + B_t[:, :, tt].T @ P_next @ B_t[:, :, tt] )

    K_t_temp =  -np.linalg.inv(M_t) @ (B_t[:, :, tt].T @ P_next @ A_t[:, :, tt])
    K_t.insert(0, K_t_temp)

    # Compute Riccati updates
    P_t = Q_t + A_t[:, :, tt].T @ P_next @ A_t[:, :, tt] - K_t_temp.T @ (R_t + B_t[:,:,tt].T @ P_next @ B_t[:,:,tt]) @ K_t_temp

    P_next = P_t


#INITIALIZATION OF THE SURFACE POSITION
xx = np.zeros((ns, TT))
uu = np.zeros((ni, TT))
err = np.zeros((ns, TT))

#xx[:, 0] = xx_ref[:, 0]
xx[:, 0] = np.array([-0.002, -0.002, -0.002, -0.002, 0, 0, 0, 0])


for tt in range(TT - 1):
    uu[:, tt] = uu_ref[:, tt] + K_t[tt] @ (xx[:, tt] - xx_ref[:, tt])
    xx[:, tt + 1] = dyn.dynamics(xx[:, tt], uu[:, tt])[0].squeeze()
    err[:, tt] = xx[:, tt] - xx_ref[:, tt]

# Save tracked trajectories for the animation
#np.save('xx_tracked.npy', xx)
#np.save('uu_tracked.npy', uu)



####################
#TRACKING ERROR ANALYSIS
####################

# Time vector for plotting
time = np.linspace(0, tf, TT)

print('Maximum position error on second half:', np.max(abs(err[:4, int((TT-1)/2):])))
print('Maximum velocity error on second half:', np.max(abs(err[4:, int((TT-1)/2):])))

print('Mean position error:', np.mean(abs(err[:4, :])))
print('Mean velocity error:', np.mean(abs(err[4:, :])))

print('Mean position error on first half:', np.mean(abs(err[:4,0:int((TT-1)/2)])))
print('Mean velocity error on first half:', np.mean(abs(err[4:, 0:int((TT-1)/2)])))

#CONVERGENCE TIME
# Parameters
threshold_percentage = 0.1  # 10%
consecutive_steps = 2000    # Consecutive steps below threshold

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
for tt in range(TT):
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

# print results
if convergence_time is not None:
    print(
        f"\nConvergence time: {convergence_time:.2f} s (threshold: {threshold_percentage * 100}% of maximum initial error)")
else:
    print("\nNo convergence in this time interval.")



######################################
#PLOT
######################################

# Time vector for plotting
time = np.linspace(0, tf, TT)

# Plot for the first four state trajectories
plt.figure(figsize=(11, 7))
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(time, xx[i, :], label=f'z{i+1}', color='blue')
    plt.plot(time, xx_ref[i, :], label=f'z{i+1}_ref', linestyle='--', color='red')
    plt.title(f'State {i+1} Trajectory', fontsize=10)
    plt.xlabel('Time [s]', fontsize=8)
    plt.ylabel(f'z{i+1}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for the second four state trajectories
plt.figure(figsize=(11, 7))
for i in range(4, 8):
    plt.subplot(4, 1, i - 3)
    plt.plot(time, xx[i, :], label=f'$\dot{{z}}_{i+1}$', color='blue')
    plt.plot(time, xx_ref[i, :], label=f'z{i+1}_ref', linestyle='--', color='red')
    plt.title(f'State {i+1} Trajectory', fontsize=10)
    plt.xlabel('Time [s]', fontsize=8) 
    plt.ylabel(f'$\dot{{z}}_{i+1}$')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.grid(True)
plt.tight_layout()
plt.show()

# Plot tracking errors
plt.figure(figsize=(10, 7))
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

# Plot the input trajectory (uu) and reference input trajectory (uu_ref)
plt.figure(figsize=(12, 6))
for i in range(ni):
    plt.plot(time, uu[i, :], label=f'u{i+1}', color='blue')
    plt.plot(time, uu_ref[i, :], label=f'u{i+1}_ref', linestyle='--', color='red')
plt.title('Input Tracking')
plt.xlabel('Time [s]', fontsize=8)
plt.ylabel('Input Values')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.grid(True)
plt.show()




