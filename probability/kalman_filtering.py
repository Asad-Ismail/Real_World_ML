import numpy as np
import matplotlib.pyplot as plt

# Simulating some data
np.random.seed(42)
true_position = np.linspace(0, 100, 100)
noisy_measurement = true_position + np.random.normal(0, 10, 100)  # simulating noisy GPS data

# Kalman filter parameters
initial_state = np.array([0, 1])
state_estimate = initial_state

state_transition = np.array([[1, 1], [0, 1]])
process_noise_covariance = np.array([[1, 0.1], [0.1, 1]])
measurement_function = np.array([[1, 0]])
measurement_noise_covariance = np.array([[10]])

state_covariance = np.array([[100, 0.1], [0.1, 5]])

estimated_positions = []

for measurement in noisy_measurement:
    # Prediction step
    predicted_state_estimate = np.dot(state_transition, state_estimate)
    predicted_cov_estimate = np.dot(state_transition, np.dot(state_covariance, state_transition.T)) + process_noise_covariance
    
    # Update step
    kalman_gain = np.dot(predicted_cov_estimate, np.dot(measurement_function.T, np.linalg.inv(np.dot(measurement_function, np.dot(predicted_cov_estimate, measurement_function.T)) + measurement_noise_covariance)))
    state_estimate = predicted_state_estimate + np.dot(kalman_gain, (measurement - np.dot(measurement_function, predicted_state_estimate)))
    state_covariance = np.dot((np.eye(2) - np.dot(kalman_gain, measurement_function)), predicted_cov_estimate)
    
    estimated_positions.append(state_estimate[0])

# Plotting the true positions, noisy measurements, and Kalman filter estimates
plt.figure(figsize=(14, 8))
plt.plot(true_position, label="True Position", color="green")
plt.plot(noisy_measurement, label="Noisy Measurements", color="red")
plt.plot(estimated_positions, label="Kalman Filter Estimates", color="blue")
plt.legend()
plt.title("Kalman Filter for Tracking a Moving Object")
plt.xlabel("Time step")
plt.ylabel("Position")
plt.grid(True)
plt.show()
