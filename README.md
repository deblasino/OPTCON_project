# OPTCON_project
Optimal control Project. 

We have a deformable surface modelled as four coupled masses, but only two of them have actuators. The task: make all four follow a desired shape over time, using two inputs to control four degrees of freedom, on nonlinear dynamics. Trajectory generation with a Newton-type optimal control method, then two tracking controllers (LQR and MPC) compared head to head under perturbations and input constraints.

Task 1 & 2 — Trajectory generation

Given a reference that steps between two equilibrium configurations, find the input sequence that drives the system along it while minimizing a quadratic cost.

Equilibria are found numerically with scipy.optimize.root on the steady-state equations.
The Newton-type method linearizes the dynamics around the current trajectory at every time step (Jacobians derived symbolically with SymPy, then evaluated numerically), solves the resulting LQ subproblem, and updates the whole input trajectory.
Armijo backtracking selects the step size. Toggleable against a fixed step and the difference in convergence is exactly the argument for having it.
Task 1 uses a non-smooth step reference, Task 2 a smooth interpolation between equilibria. The smooth reference is what makes the optimization behave well enough to be worth tracking, which is why Tasks 3 and 4 both build on Task 2's output.

############################################################

Task 3 — LQR tracking

Linearize around the optimal trajectory at each time step, solve the Difference Riccati Equation backward to get time-varying feedback gains, apply to the nonlinear system with a perturbed initial state. Errors converge quickly and control action stays smooth.

############################################################

Task 4 — MPC tracking
