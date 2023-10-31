import cv2
import numpy as np

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

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change to your video source

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
    else:
        # Marker not detected
        estimated_location = state[:2]

    # Draw the estimated location on the frame
    cv2.circle(frame, (int(estimated_location[0]), int(estimated_location[1])), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('ArUco Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

