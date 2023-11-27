import os
import csv
import matplotlib.pyplot as plt
import sys
import numpy as np

def read_trajectory_from_file(filename):
    trajectory = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        target_x, target_y = map(float, next(reader))
        header = next(reader)
        if header is not None:
            for row in reader:
                timestamp, x, y = map(float, row)
                trajectory.append(((timestamp, (x, y))))
    return trajectory, (target_x, target_y)

def visualize_trajectories(trajectories, target_points):
    plt.figure(figsize=(8, 8))

    # Set up a color map for target points
    colors = plt.cm.rainbow(np.linspace(0, 1, len(target_points)))

    # Plot trajectories
    for i, (trajectory, target_point) in enumerate(zip(trajectories, target_points)):
        timestamps, positions = zip(*trajectory)
        x, y = zip(*positions)
        plt.plot(x, y, color=colors[i], label=f'Trajectory {i + 1}')

        # Plot target point with a specific color
        plt.scatter(target_point[0], target_point[1], marker='x', color=colors[i], s=100, label=f'Target Point {i + 1}')

    plt.title('Trajectory Visualization')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()


def process_directory(directory_path):
    # Get all CSV files in the specified directory
    csv_files = [file for file in os.listdir(directory_path) if file.endswith('.csv')]

    # Read trajectories from each CSV file
    trajectories = []
    target_points = []
    for csv_file in csv_files:
        trajectory, target_point = read_trajectory_from_file(os.path.join(directory_path, csv_file))
        trajectories.append(trajectory)
        target_points.append(target_point)

    # Visualize trajectories
    visualize_trajectories(trajectories, target_points)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_trajectory.py <directory_path>")
        sys.exit(1)

    # Get the directory path from the command-line argument
    directory_path = sys.argv[1]

    # Process all CSV files in the specified directory
    process_directory(directory_path)
