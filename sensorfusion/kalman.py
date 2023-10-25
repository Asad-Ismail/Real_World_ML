import numpy as np

# Define initial state and uncertainty
x = np.array([0, 0])  # Initial position and velocity
P = np.array([[1, 0], [0, 1]])  # Initial uncertainty

# Define process noise (uncertainty in prediction)
Q = np.array([[0.1, 0], [0, 0.1]])

# Define measurement noise for camera and radar
R_camera = 0.2
R_radar = 0.1

# Define measurement matrix
H = np.array([[1, 0]])

# Define time step
dt = 1

# Define state transition matrix
A = np.array([[1, dt], [0, 1]])

# Kalman filter function
def kalman_filter(x, P, z, R):
    # Predict
    x = A @ x
    P = A @ P @ A.T + Q
    
    # Update
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    y = z - H @ x
    x = x + K @ y
    P = (np.eye(len(x)) - K @ H) @ P
    
    return x, P

# Measurements from camera and radar
z_camera = np.array([10])
z_radar = np.array([9.5])

# Sensor fusion using Kalman filter
x, P = kalman_filter(x, P, z_camera, R_camera)
x, P = kalman_filter(x, P, z_radar, R_radar)

print("Final position estimate:", x[0])
print("Final velocity estimate:", x[1])

