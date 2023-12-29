import numpy as np

# Define the Kalman filter update function
def kalman_update(prior_mean, prior_covariance, measurement, measurement_noise_covariance):
    # Kalman gain
    K = prior_covariance / (prior_covariance + measurement_noise_covariance)
    
    # Updated (posterior) estimates
    posterior_mean = prior_mean + K * (measurement - prior_mean)
    posterior_covariance = (1 - K) * prior_covariance
    
    return posterior_mean, posterior_covariance

# Initial state estimate
initial_position = 50  # hypothetical initial position estimate
initial_uncertainty = 10  # initial uncertainty

# Radar measurements and characteristics
radar_measurements = [49, 51, 50.5, 51]  # example radar measurements
radar_noise = 2  # radar measurement noise (standard deviation)

# Camera measurements and characteristics
camera_measurements = [50.5, 50, 49.5, 50]  # example camera measurements
camera_noise = 1  # camera measurement noise (standard deviation)

# Iterate through measurements
position_estimate = initial_position
position_uncertainty = initial_uncertainty

for i in range(len(radar_measurements)):
    # Update with radar measurement
    position_estimate, position_uncertainty = kalman_update(
        position_estimate, position_uncertainty, radar_measurements[i], radar_noise**2)
    
    # Update with camera measurement
    position_estimate, position_uncertainty = kalman_update(position_estimate, position_uncertainty, camera_measurements[i], camera_noise**2)
    
    print(f"Updated position estimate after step {i+1}: {position_estimate:.2f}")

