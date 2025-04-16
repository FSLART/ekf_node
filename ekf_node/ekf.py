import numpy as np
import time

class EKF(object):
    def __init__(self, initial_state, initial_covariance, noise, wheelbase):
        # Ensure initial_state is float to avoid dtype issues
        self.state = initial_state.astype(np.float64)  # [x, y, theta, v]
        self.P = initial_covariance.astype(np.float64)  # Covariance matrix
        self.noise = noise.astype(np.float64)  # Process noise
        self.wheelbase = wheelbase
        self.last_time = time.time()

    def predict(self, u):
        current_time = time.time()
        dt = current_time - self.last_time

        self.last_time = current_time

        v = u[0]  # Control input velocity (float)
        delta = u[1] #wheel angle (float)
 
        beta = np.arctan(np.tan(delta) / 2)  # slip angle (difference between wheel angle and car angle)
        theta = self.state[2, 0]

        # Update state (using floats)
        self.state[0, 0] += v * np.cos(theta + beta) * dt  # x   only beta
        self.state[1, 0] += v * np.sin(theta + beta) * dt  # y   only beta
        self.state[2, 0] += (v / self.wheelbase) * np.tan(delta) * dt  # theta i.e. Heading angle    beta
        self.state[3, 0] = v  # Velocity from control input

        # Normalize theta, maybe not needed
        self.state[2, 0] = (self.state[2, 0] + np.pi) % (2 * np.pi) - np.pi

        # Jacobian F
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = -v * np.sin(beta) * dt
        F[1, 2] = v * np.cos(beta) * dt
        F[2, 3] = 0  # Theta does not depend on previous velocity

        # Noise Jacobian G
        G = np.array([
            [np.cos(beta) * dt, 0],
            [np.sin(beta) * dt, 0],
            [(np.tan(beta) / self.wheelbase) * dt, (v / (self.wheelbase * np.cos(beta)**2) * dt)],
            [1, 0]], dtype=np.float64)

        # Process noise covariance
        R = G @ self.noise @ G.T

        # Predict covariance
        self.P = F @ self.P @ F.T + R

    def update(self, z, R):
        '''z has the same format as the state array (x,y,theta,v)'''
        z_pred = self.state.copy() #state from the prediction step
        y = z - z_pred #current state - predicted state
        H = np.eye(4, dtype=np.float64) # identity matrix because the state array is the same as the measurement
        S = H @ self.P @ H.T + R # identity * the covariance matrix * the transposed of the identity + the measurement noise
        K = self.P @ H.T @ np.linalg.inv(S) # covariance * identity transposed * inverse of the previous calculation
        self.state += K @ y  # add to the state the kalman gain * the difference between the measurement and the predicted state
        #self.state = self.state.reshape(-1, 1) #not needed, it is already a one column vector
        I = np.eye(self.P.shape[0], dtype=np.float64)# identity matrix with the same shape as the covariance matrix
        self.P = (I - K @ H) @ self.P # update the covariance matrix -> (identity - kalman gain * identity) * covariance matrix


# Example Usage with Float Initial State

# if __name__ == "__main__" :
#     initial_state = np.array([[0.0], [0.0], [0.0], [0.0]])  # Float dtype
#     initial_covariance = np.eye(4) * (0.1**2)
#     process_noise = np.diag([0.1**2, 0.1**2]).astype(np.float64)
#     wheelbase = 1.55

#     ekf = EKF(initial_state, initial_covariance, process_noise, wheelbase)

#     time.sleep(0.1)
#     control_input = np.array([1.0, 0.1], dtype=np.float64)

#     measurement = np.array([[0.16], [0.05], [0.0064732], [0.95]], dtype=np.float64)
#     measurement_noise = np.eye(4) * 0.005
#     ekf.predict(control_input)
#     print("State before update:\n", ekf.state)
#     print(f"time:{time.time()-ekf.last_time}")
#     time.sleep(0.1)
#     ekf.update(measurement, measurement_noise)
#     print("State after update:\n", ekf.state)
#     print(f"time:{time.time()-ekf.last_time}")
