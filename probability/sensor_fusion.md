- Kalman Filter Basics:
  - Used for state estimation in the presence of noise.
  - Combines prior estimate, system dynamics, control inputs, and measurements.

- Key Components:
  - State Estimate: The current best estimate of the system's state.
  - State Covariance: Uncertainty associated with the state estimate.
  - Kalman Gain: Determines how much trust to place in the measurement vs. the prior estimate.

- Sensor Fusion:
  - Combining measurements from multiple sensors to improve state estimation.
  
  - Methods of Fusion:
    * Simultaneous:
      - Combine measurements from multiple sensors before updating the state.
      - Example: Averaging or weighted combining of measurements.
    * Sequential:
      - Update state with measurements from one sensor, followed by the next.
      - Each sensor's measurement updates the state estimate independently.

  - Sequential vs Simultaneous:
    * Sequential: Offers flexibility to process sensor data as they arrive. Useful in asynchronous sensor setups.
    * Simultaneous: Can be more computationally efficient and provides a consolidated update to the state. Better for synchronized measurements.
  
  - Other Fusion Methods:
    * Weighted averaging based on sensor reliability or precision.
    * Bayesian methods to combine belief distributions.
    * Non-linear filters like Particle Filters for non-Gaussian problems.

- Example:
  - Two sensors: radar and camera.
  - Sequential updates: First update with radar, then with camera.
  - Kalman update function used for each sensor's measurements.

- Code Outline:
  1. Define Kalman update function.
  2. Initialize state estimate and uncertainty.
  3. Provide measurements and noise characteristics for each sensor.
  4. Sequentially update state using measurements from each sensor.
  5. Print the updated state estimate.

- Note:
  - Simplified example for illustration.
  - Real-world scenarios may require handling of state dynamics, control inputs, multi-dimensional states, and sensor calibrations.
