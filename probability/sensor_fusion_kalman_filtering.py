import numpy as np
import matplotlib.pyplot as plt

# Simulate true car position
np.random.seed(42)
true_position = np.linspace(0, 100, 100)

# Simulated measurements
radar_noise = np.random.normal(0, 5, 100)
camera_noise = np.random.normal(0, 7, 100)

radar_measurements = true_position + radar_noise
camera_measurements = true_position + camera_noise

# Kalman filter parameters
initial_state = 0  # Initial position estimate
state_estimate = initial_state

state_transition = np.array([[1]])
process_noise_covariance = np.array([[1]])
measurement_function = np.array([[1]])
measurement_noise_covariance = np.array([[6]])  # Average noise between radar and camera

state_covariance = np.array([[10]])

estimated_positions = []

for r, c in zip(radar_measurements, camera_measurements):
    # Take the average of radar and camera measurements
    fused_measurement = (r + c) / 2.0
    
    # Prediction step
    predicted_state_estimate = np.dot(state_transition, state_estimate)
    predicted_cov_estimate = np.dot(state_transition, np.dot(state_covariance, state_transition.T)) + process_noise_covariance
    
    # Update step
    kalman_gain = np.dot(predicted_cov_estimate, np.dot(measurement_function.T, np.linalg.inv(np.dot(measurement_function, np.dot(predicted_cov_estimate, measurement_function.T)) + measurement_noise_covariance)))
    state_estimate = predicted_state_estimate + np.dot(kalman_gain, (fused_measurement - np.dot(measurement_function, predicted_state_estimate)))
    state_covariance = np.dot((np.eye(1) - np.dot(kalman_gain, measurement_function)), predicted_cov_estimate)
    
    estimated_positions.append(state_estimate[0])

# Plot
plt.figure(figsize=(14, 8))
plt.plot(true_position, label="True Position", color="green")
plt.plot(radar_measurements, label="Radar Measurements", color="red", alpha=0.6)
plt.plot(camera_measurements, label="Camera Measurements", color="purple", alpha=0.6)
plt.plot(estimated_positions, label="Kalman Filter Estimates (Fused)", color="blue")
plt.legend()
plt.title("Sensor Fusion using Kalman Filter: Radar & Camera")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.grid(True)
plt.show()
