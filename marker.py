import cv2
import numpy as np
import time
from datetime import datetime
import random

def ekf_predict(state, P, F, Q):
    state = np.dot(F, state)
    P = np.dot(np.dot(F, P), F.T) + Q
    return state, P

def ekf_correct(state, P, H, R, measurement):
    y = measurement - np.dot(H, state)
    S = np.dot(np.dot(H, P), H.T) + R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
    state = state + np.dot(K, y)
    P = P - np.dot(np.dot(K, H), P)
    return state, P

def draw_trajectory(frame, trajectory):
    if len(trajectory) > 1:
        coordinates_array = np.array([point[1] for point in trajectory], dtype=np.int32)
        cv2.polylines(frame, [coordinates_array], isClosed=False, color=(255, 0, 0), thickness=2)

def save_trajectory_to_file(trajectory, target_point):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trajectory_{current_datetime}.csv"
    
    with open(filename, 'w') as file:
        file.write(f"TargetX, Target_Y\n")
        file.write(f"{target_point[0]}, {target_point[1]}\n")
        file.write("Timestamp, X, Y\n")
        for timestamp, coordinates in trajectory:
            file.write(f"{timestamp}, {coordinates[0]}, {coordinates[1]}\n")

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change to your video source
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video width: {width}, height: {height}, FPS: {fps}")

# ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Initialize EKF parameters
state = np.array([0, 0, 0, 0], dtype=np.float32)  # State: [x, y, vx, vy]
P = np.eye(4, dtype=np.float32)
F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=np.float32)
Q = np.eye(4, dtype=np.float32)  # Process noise covariance

# Measurement matrix (identity matrix since we directly measure the position)
H = np.eye(2, 4, dtype=np.float32)
R = np.eye(2, dtype=np.float32)  # Measurement noise covariance

# Trajectory list to store points
trajectory = []

# Generate a random target point outside the loop
target_point = (random.randint(0, int(width)), random.randint(0, int(height)))
print(f"Target point: {target_point}")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect ArUco marker
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # Marker detected
        center = np.mean(corners[0][0], axis=0)
        measurement = np.array([center[0], center[1]], dtype=np.float32)

        # Predict step
        state, P = ekf_predict(state, P, F, Q)

        # Correct step
        state, P = ekf_correct(state, P, H, R, measurement)

        estimated_location = state[:2]

        # Update trajectory with timestamp
        timestamp = time.time()
        trajectory.append((timestamp, (int(estimated_location[0]), int(estimated_location[1]))))

        # Draw trajectory on the frame
        draw_trajectory(frame, trajectory)
    else:
        # Marker not detected
        estimated_location = state[:2]

    # Draw the estimated location on the frame
    cv2.circle(frame, (int(estimated_location[0]), int(estimated_location[1])), 5, (0, 255, 0), -1)

    # Draw the fixed target point with star marker
    cv2.drawMarker(frame, target_point, (0, 0, 255), cv2.MARKER_STAR, 20, 2)

    frame = cv2.flip(frame, 1)

    # Display the frame
    cv2.imshow('ArUco Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        # Save trajectory to a file before exiting
        save_trajectory_to_file(trajectory, target_point)
        break

cap.release()
cv2.destroyAllWindows()
