import numpy as np
import time

class EKF(object):
    def __init__(self, initial_state, noise):
        #self.wheelbase = wheelbase
        
        #Time initialization
        self.last_time = time.time()

        # auxiliary variables
        self.n_state = 3 # Can be swithched for a Constant
        self.n_landmarks = 0
        self.Fx = np.eye(3)

        # Ensure initial_state is float to avoid dtype issues
        self.state = initial_state.astype(np.float64)  # [x, y, theta]
        self.P = np.zeros((self.n_state+2*self.n_landmarks,self.n_state+2*self.n_landmarks)) # Covariance matrix
        np.fill_diagonal(self.P,100) # Initialize state uncertainty with large variances, no correlations
        self.R = noise.astype(np.float64)  # Process noise


    def predict(self, v, w):

        # Getting the time difference
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Getting the state
        theta = self.state[2, 0]  # Robot heading
        
        # Update state estimate mu with model
        state_model_mat = np.zeros((3,1)) # Initialize state update matrix from model

        state_model_mat[0] = -(v/w)*np.sin(theta)+(v/w)*np.sin(theta+w*dt) if np.abs(w)>0.01 else v*np.cos(theta)*dt # Update in the robot x position
        state_model_mat[1] = (v/w)*np.cos(theta)-(v/w)*np.cos(theta+w*dt) if np.abs(w)>0.01 else v*np.sin(theta)*dt # Update in the robot y position
        state_model_mat[2] = w*dt # Update for robot heading theta

        # Update the state
        self.state = self.state + np.matmul(np.transpose(self.Fx),state_model_mat) # Update state estimate, simple use model with current state estimate
        self.state[2] = (self.state[2] + np.pi) % (2 * np.pi) - np.pi # Normalize theta to be between -pi and pi

        # Jacobian F
        state_jacobian = np.zeros((self.n_state,self.n_state)) # Initialize model jacobian
        state_jacobian[0,2] = -(v/w)*np.cos(theta) + (v/w)*np.cos(theta+w*dt) if np.abs(w)>0.01 else -v*np.sin(theta)*dt # Jacobian element, how small changes in robot theta affect robot x
        state_jacobian[1,2] = -(v/w)*np.sin(theta) + (v/w)*np.sin(theta+w*dt) if np.abs(w)>0.01 else v*np.cos(theta)*dt # Jacobian element, how small changes in robot theta affect robot y
        
        G = np.eye(self.P.shape[0]) + np.transpose(self.Fx).dot(state_jacobian).dot(self.Fx) # How the model transforms uncertainty

        self.P = G.dot(self.P).dot(np.transpose(G)) + np.transpose(self.Fx).dot(self.R).dot(self.Fx) # Combine model effects and stochastic noise    

        
        '''
        # Jacobian F
        F = np.eye(3, dtype=np.float64)
        F[0, 2] = -vx * np.sin(theta) * dt
        F[1, 2] = vy * np.cos(theta) * dt

        # Noise Jacobian G
        G = np.array([
        [np.cos(theta) * dt, 0], 
        [np.sin(theta) * dt, 0],
        [0, dt]
        ])
        
        # Predict covariance
        self.P = (F @ self.P @ F.T) + (G @ self.noise @ G.T)
        '''
    
        

    def update(self, z, R):
        '''z has the same format as the state array (x,y,theta,v)'''
        z_pred = self.state.copy() #state from the prediction step
        y = z - z_pred #current state - predicted state
        H = np.eye(3, dtype=np.float64) # identity matrix because the state array is the same as the measurement # USED TO BE 4!
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
