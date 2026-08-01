Optimal Control Project 
Group 42 - Alessandro De Blasi, Kavishe Kewalramani, Simone Romeo



TASK 1: trajectory generation - non-smooth

BRIEF DESCRIPTION:

This task implements an optimal control strategy for an under-actuated flexible surface system using a Newton-type method with Armijo step size selection. 
The system involves two actuated points on a surface, modeled with nonlinear dynamics. The goal is to compute control inputs that drive the system along a desired reference trajectory while minimizing a quadratic cost function.


CODE OVERVIEW:

-main_newton.py
Main script for running the optimal control algorithm. Generates reference trajectories, iteratively optimizes control inputs, and visualizes results.

-dynamics_def.py
Symbolic and numerical implementation of the system dynamics, including Jacobian computations for linearization.

-reference_trajectory.py
Generates reference trajectories by computing equilibrium points and transitioning between them.

-eq_try_2.py
Attempts to compute equilibrium states and inputs using scipy.optimize.root.

-armijo.py
Implements the Armijo rule for adaptive step size selection during gradient descent optimization.

-cost.py
Defines quadratic stage and terminal cost functions for tracking state and input references.


BEFORE RUNNING - PARAMETERS TO ADJUST (OPTIONAL):

-main_newton.py

	max_iters: maximum algorithm iterations.

	Armijo: if True, toggle Armijo step size selection. Otherwise, use fixed_stepsize.

	cc and beta: parameters for Armijo.

	term_cond: if the cost reaches this value, the algorithm stops.

	visu_descent_plot: if True, show cost and gradient descent plots.

	tf: simulation duration.

-cost.py

	QQt: diagonal weights for state penalization. First 4 elements relative to positions, last 4 relative to velocities.

	RRt: diagonal cost matrix for input penalization

-dynamics_def.py 

	dt: discretization step-size used for forward Euler method




HOW TO RUN THE Task:

Run main_newton.py





TASK 2: trajectory generation - smooth

BRIEF DESCRIPTION:

This task implements an optimal control strategy for an under-actuated flexible surface system using a Newton-type method with Armijo step size selection. 
The system involves two actuated points on a surface, modeled with nonlinear dynamics. The goal is to compute control inputs that drive the system along a desired smooth reference trajectory while minimizing a quadratic cost function.



CODE OVERVIEW:

-main_newton.py
Main script for running the optimal control algorithm. Generates reference trajectories, iteratively optimizes control inputs, and visualizes results.

-dynamics_def.py
Symbolic and numerical implementation of the system dynamics, including Jacobian computations for linearization.

-smooth_traj_gen.py
Generates smooth reference trajectories by computing equilibrium points and interpolating them.

-eq_try_2.py
Attempts to compute equilibrium states and inputs using scipy.optimize.root.

-armijo.py
Implements the Armijo rule for adaptive step size selection during gradient descent optimization.

-cost.py
Defines quadratic stage and terminal cost functions for tracking state and input references.


BEFORE RUNNING - PARAMETERS TO ADJUST (OPTIONAL):

-main_newton.py

	max_iters: maximum algorithm iterations.

	Armijo: if True, toggle Armijo step size selection. Otherwise, use fixed_stepsize.

	cc and beta: parameters for Armijo.

	term_cond: if the cost reaches this value, the algorithm stops.

	visu_descent_plot: if True, show cost and gradient descent plots.

	tf: simulation duration.

-cost.py

	QQt: diagonal weights for state penalization. First 4 elements relative to positions, last 4 relative to velocities.

	RRt: diagonal cost matrix for input penalization

-dynamics_def.py 

	dt: discretization step-size used for forward Euler method

-smooth_traj_gen.py
	
	positions: it is possible to change the positions to find other equilibrium points


HOW TO RUN THE TASK:

Run main_newton.py





TASK 3: trajectory tracking - LQR


BRIEF DESCRIPTION:

This task implements a Linear Quadratic Regulator (LQR) for tracking a reference trajectory in a dynamical system (under-actuated flexible surface system). It computes optimal control gains to minimize tracking errors while balancing state deviations and control effort. The system's dynamics and reference trajectory are loaded from files (precomputed from TASK 2 and stored in a .npy file). 
In the end, results are visualized via state, input, and error plots.


CODE OVERVIEW:

-LQR_tracking.py
Main script for running the tracking algorithm. Computes optimal gains and visualizes results.

-dynamics_def.py
Symbolic and numerical implementation of the system dynamics, including Jacobian computations for linearization.


TRAJECTORY GENERATION:

The trajectory generation has been done in TASK 2 and saved into 'xx_star_1e-4_1s_def.npy' and 'uu_star_1e-4_1s_def.npy'. Those files have been imported in TASK 3 to speed up the computation.
IMPORTANT - ensure these files are in the same folder of the code.


BEFORE RUNNING - PARAMETERS TO ADJUST (OPTIONAL):

-LQR_tracking.py

	tf: simulation duration.

	Q_t: diagonal weights for state penalization. First 4 elements relative to positions, last 4 relative to velocities.

	R_t: diagonal cost matrix for input penalization

	xx[:, 0]: initial perturbation of the state. Set each value to 0 to remove the perturbation.

-dynamics_def.py 

	dt: discretization step-size used for forward Euler method


HOW TO RUN THE TASK:

Run LQR_tracking.py





TASK 4: trajectory tracking - MPC

BRIEF DESCRIPTION:

This task implements a Model Predictive Control (MPC) for tracking reference trajectories on a linear system with constraints. The code solves a constrained optimization problem to compute optimal control inputs while adhering to state and input constraints. The system's dynamics and reference trajectory are loaded from files (precomputed from TASK 2 and stored in a .npy file). 
In the end, results are visualized via state, input, and error plots.


CODE OVERVIEW:

-main_mpc.py
Main script for running the tracking algorithm. Computes optimal control inputs and visualizes results.

-dynamics_def.py
Symbolic and numerical implementation of the system dynamics, including Jacobian computations for linearization.

-solver.py
Script for solving a QP problem using cvxpy


TRAJECTORY GENERATION:

The trajectory generation has been done in TASK 2 and saved into 'xx_star_1e-4_1s_def.npy' and 'uu_star_1e-4_1s_def.npy'. Those files have been imported in TASK 3 to speed up the computation.
IMPORTANT - ensure these files are in the same folder of the code.


BEFORE RUNNING - PARAMETERS TO ADJUST (OPTIONAL):

-main_mpc.py

	tf: simulation duration.

	T_pred: prediction horizon for the MPC

	Q_t: diagonal weights for state penalization. First 4 elements relative to positions, last 4 relative to velocities.

	R_t: diagonal cost matrix for input penalization

	umax, umin: constraints on the input

	QP_solve_freq: controls how frequently the QP is solved. If set to 1, every QP problem is solved. The more QP problem are solved, the slower the computation (but also more accurate)

	xx0: initial perturbation of the state. Set each value to 0 to remove the perturbation.

-dynamics_def.py 

	dt: discretization step-size used for forward Euler method

-solver.py

	solver (in problem.solve): OSQP solver has been selected, but it may be changed if needed.

	warm_start (in problem.solve): set to False by default because it was giving better results. May eventually be set to True if needed.


HOW TO RUN THE TASK:

Run main_mpc.py





