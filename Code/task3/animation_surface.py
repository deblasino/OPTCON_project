from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import dynamics_def as dyn

def animate_surface(xx_star, xx_ref, dt):
    TT = xx_star.shape[1]  # Number of time steps
    num_points = 4  # Number of points on the surface
    spacing = 0.3  # Distance between points

    # Add two stationary points (one at the beginning and one at the end)
    total_points = num_points + 2  # Including stationary points
    x_positions = np.arange(0, total_points) * spacing  # Positions of all points

    # Set up the figure and axis for the animation
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5 * spacing, (total_points - 0.5) * spacing)  # Adjust for spacing and additional points
    ax.set_ylim(-0.02, 0.02)  # Vertical limits for displacement

    # Plot elements
    optimal_line, = ax.plot([], [], 'o-', lw=3, color="blue", label="Optimal Path")
    reference_line, = ax.plot([], [], 'o--', lw=2, color="green", label="Reference Path")
    stationary_points, = ax.plot([], [], 'ro', markersize=10, label="Stationary Points")  # Red stationary points

    time_text = ax.text(0.5, 0.1, '', transform=ax.transAxes, ha='center')

    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.15))
    ax.set_title("Surface animation")
    ax.set_xlabel("Position")
    ax.set_ylabel("Vertical Displacement")

    # Initialization function for the animation
    def init():
        optimal_line.set_data([], [])
        reference_line.set_data([], [])
        stationary_points.set_data([], [])
        time_text.set_text('')
        return optimal_line, reference_line, stationary_points, time_text

    # update function for each frame of the animation
    def update(frame):
        # Sample every 100 point to reduce the number of frames
        frame_index = frame * 100
        if frame_index >= TT:
            frame_index = TT - 1

        # Extract displacements for all points at the current time step
        z_opt = xx_star[:num_points, frame_index]
        z_ref = xx_ref[:num_points, frame_index]

        # Add zero displacement for the stationary points
        z_opt_full = np.concatenate(([0], z_opt, [0]))
        z_ref_full = np.concatenate(([0], z_ref, [0]))

        # Update the lines with new data
        optimal_line.set_data(x_positions, z_opt_full)
        reference_line.set_data(x_positions, z_ref_full)

        # Update stationary points (always at zero displacement)
        stationary_points.set_data([x_positions[0], x_positions[-1]], [0, 0])

        # Update time text
        time_text.set_text(f'time = {frame_index * dt:.2f}s')

        return optimal_line, reference_line, stationary_points, time_text

    # Create the animation with a reduced number of frames
    num_frames = TT // 1  # Reduce the number of frames
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=100)  # 100ms per frame

    # Display the animation
    plt.show()

# Load trajectories
xx = np.load('xx_tracked.npy')
xx_ref = np.load('xx_star_1e-4_1s_def.npy')
dt = dyn.dt

animate_surface(xx, xx_ref, dt)