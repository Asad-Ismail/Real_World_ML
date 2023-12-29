import numpy as np
import matplotlib.pyplot as plt

# Simulating data
np.random.seed(42)  # for reproducibility

# True distance to the object
true_distance = 100.0

# Number of measurements
N = 1000

# Simulated radar and camera measurements with noise
radar_noise_stddev = 2.0  # standard deviation of radar noise
camera_noise_stddev = 1.5  # standard deviation of camera noise

radar_measurements = true_distance + np.random.randn(N) * radar_noise_stddev
camera_measurements = true_distance + np.random.randn(N) * camera_noise_stddev

# Plotting the simulated data
plt.figure(figsize=(12, 6))
plt.plot(radar_measurements, label="Radar Measurements")
plt.plot(camera_measurements, label="Camera Measurements")
plt.axhline(true_distance, color='red', linestyle='--', label="True Distance")
plt.legend()
plt.xlabel("Measurement Number")
plt.ylabel("Distance (meters)")
plt.title("Simulated Radar and Camera Measurements")
plt.show()

# Calculating covariance
measurements_matrix = np.vstack((radar_measurements, camera_measurements))
measurement_noise_covariance = np.cov(measurements_matrix)

print("Measurement Noise Covariance:")
print(measurement_noise_covariance)

'''
In the context of our simulated radar and camera distance measurements, the covariance between radar and camera measurements represents the degree to which the two sets of measurements change together.

Specifically:

Positive Covariance: If the covariance is positive, it means that when the radar reading is higher (or lower) than its mean value, the camera reading also tends to be higher (or lower) than its mean. This could suggest that some external factor is causing both sensors to read higher or lower values simultaneously.

Negative Covariance: A negative covariance would indicate that when the radar reading is higher than its mean value, the camera reading tends to be lower than its mean, and vice versa. This might suggest an inverse relationship in their errors.

Covariance Close to Zero: If the covariance is close to zero, it suggests that there's no consistent relationship in the fluctuations of the two sensors. The errors or noises in the two measurements are independent of each other.

In sensor fusion, understanding the covariance between different sensors can be crucial. If two sensors have high positive covariance, it might mean they are both susceptible to the same kind of interference or error source. If they have a negative covariance, one sensor's overestimations might be offset by the other's underestimations, and vice versa. If their covariance is near zero, the two sensors are providing independent information about the measured quantity, which can be beneficial in fusion as it can lead to more robust and accurate combined estimates.
'''





