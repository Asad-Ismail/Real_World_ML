# Kalman Filter Equations

## Prediction Step

### 1. State Prediction
\[ \hat{x}_{k|k-1} = F_k x_{k-1|k-1} + B_k u_k \]
- \( \hat{x}_{k|k-1} \) is the predicted state estimate at time \( k \) given all observations up to time \( k-1 \).
- \( F_k \) is the state transition model which is applied to the previous state \( x_{k-1|k-1} \).
- \( B_k \) is the control input model which is applied to the control vector \( u_k \).

### 2. Covariance Prediction
\[ P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k \]
- \( P_{k|k-1} \) is the predicted state covariance at time \( k \).
- \( Q_k \) is the process noise covariance which accounts for the uncertainty in the prediction.

## Update Step

### 1. Innovation or Measurement Residual
\[ y_k = z_k - H_k \hat{x}_{k|k-1} \]
- \( y_k \) is the innovation or measurement residual.
- \( z_k \) is the actual measurement at time \( k \).
- \( H_k \) is the observation model which maps the true state space into the observed space.

### 2. Innovation (or Residual) Covariance
\[ S_k = H_k P_{k|k-1} H_k^T + R_k \]
- \( S_k \) is the innovation covariance.
- \( R_k \) is the measurement noise covariance which accounts for the uncertainty in the observation.

### 3. Optimal Kalman Gain
\[ K_k = P_{k|k-1} H_k^T S_k^{-1} \]
- \( K_k \) is the Kalman gain which minimizes the a posteriori error covariance.

### 4. State Update
\[ x_{k|k} = \hat{x}_{k|k-1} + K_k y_k \]
- \( x_{k|k} \) is the a posteriori state estimate at time \( k \).

### 5. Covariance Update
\[ P_{k|k} = (I - K_k H_k) P_{k|k-1} \]
- \( P_{k|k} \) is the a posteriori error covariance at time \( k \).

