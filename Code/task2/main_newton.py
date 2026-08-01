
import numpy as np
import matplotlib.pyplot as plt

import dynamics_def as dyn
import cost as cst
import armijo
import smooth_traj_gen as ref_gen


plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams.update({'font.size': 22})

#######################################
# Algorithm parameters
#######################################

max_iters = 20
fixed_stepsize = 1e-6
term_cond = 1e-6


# ARMIJO PARAMETERS
Armijo = True
stepsize_0 = 0.7
cc = 0.5
beta = 0.7
armijo_maxiters = 20  # number of Armijo iterations
visu_descent_plot = False #to visualize armijo graph


#######################################
# Trajectory parameters
#######################################

tf = 1  # final time in seconds

dt = dyn.dt  # get discretization step from dynamics
ns = dyn.ns
ni = dyn.ni

TT = int(tf / dt)  # discrete-time samples

######################################
# Reference curve
######################################

#generating the complete desired trajectory for the state (position) and the input u
xx_ref, uu_ref = ref_gen.generate_reference_trajectory(tf=tf, dt=dt)

x0 = xx_ref[:, 0]
print(f"dimension of x0 is {x0.shape}")

######################################
# Arrays to store data
######################################

xx = np.zeros((ns, TT, max_iters))  # state seq.
uu = np.zeros((ni, TT, max_iters))  # input seq.
xx_intermediate = np.zeros((ns, TT, 3)) #to plot 3 intermediate trajectories


deltau = np.zeros((ni, TT, max_iters))  # Du - descent direction
dJ = np.zeros((ni, TT, max_iters))  # DJ - gradient of J wrt u to compute descent_arm

JJ = np.zeros(max_iters)  # collect cost

descent = np.zeros(max_iters)  # collect descent direction to plot
descent_arm = np.zeros(max_iters)  # collect descent direction to compute armijo stepsize


######################################
# MAIN
######################################

print('-*-*-*-*-*-')

kk = 0

# Create a figure and subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 8))


for kk in range(max_iters-1 ):


    #Evaluate the cost for iteration k
    JJ[kk] = 0
    # calculate cost
    for tt in range(TT - 1):

        temp_cost = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[0]
        JJ[kk] += temp_cost

    temp_cost = cst.termcost(xx[:, -1, kk], xx_ref[:, -1])[0]
    JJ[kk] += temp_cost
    print(f" Cost : {JJ[kk]}")



    ###########################
    #INITIALIZATION FOR EACH K
    ############################

    # Initialize Riccati matrices for the augmented system
    P_next = cst.QQT  # initialized with QT
    p_next =cst.termcost(xx[:, TT - 1, kk], xx_ref[:, TT - 1])[1]  # pT initialized as qT

    K_t = []
    sigma_t = []

    A_t = np.zeros((ns, ns, TT))
    B_t = np.zeros((ns, ni, TT))

    lambdas = np.zeros((ns,TT))
    lambdas[:, TT-1] = cst.termcost(xx[:, -1, kk], xx_ref[:, -1])[1].reshape(-1)


    #REVERSE LOOP TO COMPUTE K_t AND SIGMA THROUGH RICCATI AND AFFINE LQR
    for tt in reversed(range(TT-1)):

        #gradients of cost
        qt, rt = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[1:3]


        # (jacobians) of dynamics
        dfx, dfu = dyn.dynamics(xx[:, tt, kk], uu[:, tt, kk])[1:3]


        #hessians of cost
        hlxx, hluu, hlux = cst.stagecost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt])[3:6]

        # linearization of the system on the trajectory
        A_t[:, :, tt] = dfx
        B_t[:, :, tt] = dfu

        #compute lambdas and dJu for each t, only for armijo because we don't need the hessian of dynamics because of regularization
        lambdas[:,tt] = A_t[:, :, tt].T @ lambdas[:,tt+1] +qt
        dJ[:,tt,kk] =  B_t[:, :, tt].T @ lambdas[:,tt+1] + rt


        #Compute matrices to solve LQR problem to find the descending direction, using regularization techniques to not consider the hessian of dynamics
        Q_t = cst.QQt
        R_t = cst.RRt
        S_t = np.zeros((ni, ns))

        M_t = R_t + B_t[:, :, tt].T @ P_next @ B_t[:, :, tt] + np.eye(ni) * (np.eye(ni) * 1e-6)

        #compute k and sigma for each time t
        K_t_temp = -np.linalg.inv(M_t) @ (S_t + B_t[:,:,tt].T @ P_next @ A_t[:,:,tt])
        K_t.insert(0, K_t_temp)

        sigma_t_temp = -np.linalg.inv(R_t + B_t[:,:,tt].T @ P_next @ B_t[:,:,tt]) @ (rt + B_t[:,:,tt].T @ p_next)
        sigma_t.insert(0, sigma_t_temp)


        # Compute Riccati updates
        P_t = Q_t + A_t[:,:,tt].T @ P_next @ A_t[:,:,tt] - K_t_temp.T @ (R_t + B_t[:,:,tt].T @ P_next @ B_t[:,:,tt]) @ K_t_temp
        p_t = qt + A_t[:,:,tt].T @ p_next - K_t_temp.T @ (R_t + B_t[:,:,tt].T @ P_next @ B_t[:,:,tt]) @ sigma_t_temp

        # Update Riccati variables for next iteration
        P_next = P_t
        p_next = p_t


    ############################
    # ARMIJO STEPSIZE SELECTION
    ############################

    #compute descending direction deltau (the whole vector at k step for each t) to compute armijo stepsize
    # initialize deltax
    delta_x = np.zeros((ns, TT))
    delta_x[:, 0] = 0  # deltax_0 is always zero

    for tt in range(TT - 1):
        deltau[:,tt,kk] = K_t[tt] @ delta_x[:,tt] + sigma_t[tt]
        delta_x[:,tt+1] = A_t[:,:,tt]@ delta_x[:,tt] + B_t[:,:,tt]@deltau[:,tt,kk]

        descent[kk] += deltau[:, tt, kk].T @ deltau[:, tt, kk]
        descent_arm[kk] += dJ[:, tt, kk].T @ deltau[:, tt, kk]


    if Armijo:

        stepsize = armijo.select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                                           deltau[:, :, kk], xx_ref, uu_ref, x0,
                                           uu[:, :, kk],xx[:,:,kk],K_t,sigma_t, JJ[kk], descent_arm[kk], visu_descent_plot)
    else:
        stepsize = fixed_stepsize




    ############################
    # Update the current solution
    ############################

    # ONCE I HAVE KT AND SIGMAT, COMPUTE the new input and output at k+1


    xx[:, 0, kk + 1] = x0  # each x_k start from the same initial condition
    for tt in range(TT - 1):
        uu[:, tt, kk + 1] = uu[:, tt, kk]+K_t[tt] @ (xx[:, tt, kk + 1] - xx[:, tt, kk]) + (stepsize * sigma_t[tt])
        xx[:, tt + 1, kk + 1] = dyn.dynamics(xx[:, tt, kk + 1], uu[:, tt, kk + 1])[0].squeeze()

        # save the trajectories computed in the first iterations of the method to plot them
    if kk < 3:
        xx_intermediate[:, :, kk] = xx[:, :, kk]

    uu[:, -1, kk + 1] =  uu[:, -2, kk + 1]

    print(f"Descent: {descent[kk]}")
    if descent[kk] <= term_cond:
        max_iters = kk

        break

    #########################################################
    # PLOT at each iteration#
    ########################################################
    """
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

    # Plot x[1] and x_ref[1] on the first subplot
    axs[0].plot(np.linspace(0, tf, TT), xx[0, :, kk], "b-", label="z_{1}")
    axs[0].plot(np.linspace(0, tf, TT), xx_ref[0, :], "b--", label="$z_{ref}[0]$")
    axs[0].grid()
    axs[0].legend()
    axs[0].set_title("Comparison of $z_{1}$ and $z_{1ref}$")

    # Plot x[1] and x_ref[1] on the second subplot
    axs[1].plot(np.linspace(0, tf, TT), xx[1, :, kk], "c-", label="z_{2}")
    axs[1].plot(np.linspace(0, tf, TT), xx_ref[1, :], "c--", label="$z_{2ref}$")
    axs[1].grid()
    axs[1].legend()
    axs[1].set_title("Comparison of $z_{2}$ and $z_{2ref}$")

    axs[2].plot(np.linspace(0, tf, TT), uu[0, :, kk], "r-", label="u")
    axs[2].plot(np.linspace(0, tf, TT), uu_ref[0, :], "r--", label="$u_{ref}$")
    axs[2].grid()
    axs[2].legend()
    axs[2].set_title("Comparison of $u$ and $u_{ref}$")

    # Add labels and adjust layout
    for ax in axs:
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.pause(1e-1)
    """
    ############################
    # Termination condition
    ############################

    print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk, descent[kk], JJ[kk]))

    if descent[kk] <= term_cond:
        max_iters = kk

        break

xx_star = xx[:, :, max_iters - 1]
uu_star = uu[:, :, max_iters - 1]
uu_star[:, -1] = uu_star[:, -2]  # for plotting purposes

# Save optimal trajectories in .npy
#np.save('xx_star_1e-4_1s_def.npy', xx_star)
#np.save('uu_star_1e-4_1s_def.npy', uu_star)


############################
# Plots
############################

# cost and descent

plt.figure('descent direction')
plt.plot(np.arange(max_iters), descent[:max_iters])
plt.xlabel('$k$')
plt.ylabel('||$d^k||$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

plt.figure('cost')
plt.plot(np.arange(max_iters), JJ[:max_iters])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

# optimal trajectory

tt_hor = np.linspace(0, tf, TT)


for i in range(4):  # positions
    plt.figure(figsize=(14, 7))

    # reference
    plt.plot(tt_hor, xx_ref[i, :], 'r--', label='Desired Trajectory', linewidth=2)

    # intermediate
    for kk in range(3):
        plt.plot(tt_hor, xx_intermediate[i, :, kk], label=f'Iteration {kk}', linewidth=1.5)

    # optimal
    plt.plot(tt_hor, xx_star[i, :], 'b-', label='Optimal trajectory', linewidth=2.5)
    plt.xlabel('Time [s]')
    plt.ylabel(f'$z_{i + 1}$')
    plt.legend(fontsize='small')
    plt.grid()
    plt.show()


plt.show()






