import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the arm's link lengths and colors as NumPy arrays
link_lengths = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
link_colors = ['r', 'g', 'b', 'c', 'm', 'y']

# Initialize joint angles (in radians) as a NumPy array
joint_angles = np.zeros(6)

# Specify the target position 
target_x = 10.0
target_y = 10.0

# Define the tolerance for convergence
tolerance = 0.01

# Maximum number of iterations for convergence
max_iterations = 1000

# Create a function to update the arm's position
def update_arm(i):
    global joint_angles

    # Calculate end effector position using forward kinematics
    end_effector_x = 0
    end_effector_y = 0
    end_effector_positions = []

    for i in range(len(link_lengths)):
        end_effector_x += link_lengths[i] * np.cos(np.sum(joint_angles[:i+1]))
        end_effector_y += link_lengths[i] * np.sin(np.sum(joint_angles[:i+1]))
        end_effector_positions.append((end_effector_x, end_effector_y))

    # Plot the arm
    plt.clf()
    for i in range(len(link_lengths)):
        if i == 0:
            plt.plot([0, end_effector_positions[i][0]], [0, end_effector_positions[i][1]], 'k-')
        else:
            plt.plot([end_effector_positions[i-1][0], end_effector_positions[i][0]],
                     [end_effector_positions[i-1][1], end_effector_positions[i][1]], f'{link_colors[i-1]}-')

    plt.plot(target_x, target_y, 'ro')  # Mark the target position
    plt.xlim(-25, 25)  # Set X-axis limits to -25 to 25
    plt.ylim(-25, 25)  # Set Y-axis limits to -25 to 25
    plt.gca().set_aspect('equal', adjustable='box')

    # Inverse Kinematics Calculation
    end_effector_error = np.sqrt((target_x - end_effector_x)**2 + (target_y - end_effector_y)**2)

    if end_effector_error > tolerance:
        jacobian = np.zeros((2, 6))
        for i in range(len(link_lengths)):
            jacobian[0, i] = -np.sum(link_lengths[i:] * np.sin(np.sum(joint_angles[i:])))
            jacobian[1, i] = np.sum(link_lengths[i:] * np.cos(np.sum(joint_angles[i:])))
        
        jacobian_pseudo_inv = np.linalg.pinv(jacobian)
        error_x = target_x - end_effector_x
        error_y = target_y - end_effector_y
        d_theta = np.dot(jacobian_pseudo_inv, np.array([error_x, error_y]))
        joint_angles += d_theta

# Create an animation of the arm's movement
ani = FuncAnimation(plt.gcf(), update_arm, interval=100)

plt.show()