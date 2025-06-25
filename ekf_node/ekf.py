import numpy as np
from sklearn.neighbors import KDTree
import time

class EKF(object):
    def __init__(self, initial_state, noise):
        #self.wheelbase = wheelbase
        
        #Time initialization
        self.last_time = time.time()

        # auxiliary variables
        self.n_state = 3
        self.Fx = np.eye(3)

        #Landmarks
        self.blue_cones_indeces = {}
        self.yellow_cones_indeces = {}
        self.orange_cones_indeces = {}
        self.orange_big_cones_indeces = {}
        n_landmarks = 0

        # Ensure initial_state is float to avoid dtype issues
        self.state = initial_state.astype(np.float64)  # [x, y, theta]
        self.P = np.zeros((self.n_state+2*n_landmarks,self.n_state+2*n_landmarks)) # Covariance matrix
        np.fill_diagonal(self.P,100) # Initialize state uncertainty with large variances, no correlations
        self.R = noise.astype(np.float64)  # Process noise
        self.Q = np.diag([0.003,0.003]) # sigma_r, sigma_phi

    def get_cones_from_map(self, state_array, color):
        '''
        cones: global position of the cones
        color: index of the cones of a certain color
        '''
        cones = []
        for i in color:
            cone = (state_array[self.n_state+2*i,0], state_array[self.n_state+2*i+1,0])
            cones.append(cone)
        return cones

    def data_association(cones, observations, threshold=0.8):
        '''
        Perform data association between cones and observations.
        
        Args:
            cones (list): List of cone positions (2D coordinates).
            observations (list): List of observed positions (2D coordinates).
            threshold (float): Max distance for association.
        
        Returns:
            array: array the same size as observations, for each observation it returns the index of the landmark that is already in the map, -1 if its not there yet
        '''
        
        matched_cones = []
        tree = KDTree(cones)

        for obs in observations:
            indices = tree.query_radius([obs], r=threshold)[0]
            if len(indices) > 0:
                matched_cones.append(indices[0]) 
            else:
                matched_cones.append(-1)
            
        return matched_cones
    

    def data_augmentation(self, new_blue_cones, new_yellow_cones, new_orange_cones, new_orange_big_cones):

        # Define mapping of cones to color label and tracking list
        cone_sets = [
            (new_blue_cones, self.blue_cones_indices),
            (new_yellow_cones, self.yellow_cones_indices),
            (new_orange_cones, self.orange_cones_indices),
            (new_orange_big_cones, self.orange_big_cones_indices)
        ]

        for new_cone_list, cone_index_list in cone_sets:
            for cone_coords in new_cone_list:
                landmark_x, landmark_y = cone_coords  # Already in global frame

                # Add landmark to state
                new_landmark = np.array([[landmark_x], [landmark_y]])
                self.state = np.vstack((self.state, new_landmark))

                # Expand covariance matrix with high initial uncertainty
                landmark_cov = np.eye(2) * 1e3
                top_right = np.zeros((self.P.shape[0], 2))
                bottom_left = np.zeros((2, self.P.shape[1]))
                self.P = np.block([
                    [self.P,        top_right],
                    [bottom_left,   landmark_cov]
                ])

                # Save this landmark's index
                cone_index_list.append(self.n_landmarks)
                self.n_landmarks += 1


    def predict(self, v, w):

        # Getting the time difference
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Getting the state
        theta = self.state[2, 0]  # Robot heading
        
        # Update state estimate with model
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

    def update(self,z):
        
        ### DATA ASSOCIATION ###

        map_blue_cones = self.get_cones_from_map(self, self.state, self.blue_cones_indeces)
        map_yellow_cones = self.get_cones_from_map(self, self.state, self.yellow_cones_indeces)
        map_orange_cones = self.get_cones_from_map(self, self.state, self.orange_cones_indeces)
        map_orange_big_cones = self.get_cones_from_map(self, self.state, self.orange_big_cones_indeces)

        #separates the cones in the map by color
        blue_cones_converted_predicted_pose = {}
        yellow_cones_converted_predicted_pose = {}
        orange_cones_converted_predicted_pose = {}
        orange_big_cones_converted_predicted_pose = {}
        pose_x, pose_y, pose_theta = self.state[0], self.state[1], self.state[2]

        for i,obs in enumerate(z):
            obs_x, obs_y = obs.position.x, obs.position.y
            color = obs.class_type
            # Calculate the expected observation #TODO:review
            expected_obs_x = pose_x + np.cos(pose_theta) * obs_x - np.sin(pose_theta) * obs_y
            expected_obs_y = pose_y + np.sin(pose_theta) * obs_x + np.cos(pose_theta) * obs_y
            if color == 1:
                yellow_cones_converted_predicted_pose[(expected_obs_x, expected_obs_y)] = i
            elif color == 2:
                blue_cones_converted_predicted_pose[(expected_obs_x, expected_obs_y)] = i
            elif color == 3:
                orange_cones_converted_predicted_pose[(expected_obs_x, expected_obs_y)] = i
            elif color == 4:
                orange_big_cones_converted_predicted_pose[(expected_obs_x, expected_obs_y)] = i

        #gets the global position of the cones
        yellow_cones_converted_predicted_pose_keys = np.array(list(yellow_cones_converted_predicted_pose.keys()))
        blue_cones_converted_predicted_pose_keys = np.array(list(blue_cones_converted_predicted_pose.keys()))
        orange_cones_converted_predicted_pose_keys = np.array(list(orange_cones_converted_predicted_pose.keys()))
        orange_big_cones_converted_predicted_pose_keys = np.array(list(orange_big_cones_converted_predicted_pose.keys()))

        #perform data association
        matched_yellow_cones = self.data_association(map_yellow_cones, yellow_cones_converted_predicted_pose_keys, threshold=0.5)
        matched_blue_cones = self.data_association(map_blue_cones, blue_cones_converted_predicted_pose_keys, threshold=0.5)
        matched_orange_cones = self.data_association(map_orange_cones, orange_cones_converted_predicted_pose_keys, threshold=0.5)
        matched_orange_big_cones = self.data_association(map_orange_big_cones, orange_big_cones_converted_predicted_pose_keys, threshold=0.5)
        new_yellow_cones = []
        new_blue_cones = []
        new_orange_cones = []
        new_orange_big_cones = []

        for i, cords in enumerate(yellow_cones_converted_predicted_pose_keys):
            if matched_yellow_cones[i] == -1:
                new_yellow_cones.append(yellow_cones_converted_predicted_pose[tuple(cords)])

        for i, cords in enumerate(blue_cones_converted_predicted_pose_keys):
            if matched_blue_cones[i] == -1:
                new_blue_cones.append(blue_cones_converted_predicted_pose[tuple(cords)])

        for i, cords in enumerate(orange_cones_converted_predicted_pose_keys):
            if matched_orange_cones[i] == -1:
                new_orange_cones.append(orange_cones_converted_predicted_pose[tuple(cords)])

        for i, cords in enumerate(orange_big_cones_converted_predicted_pose_keys):
            if matched_orange_big_cones[i] == -1:
                new_orange_big_cones.append(orange_big_cones_converted_predicted_pose[tuple(cords)])


        ### MEASUREMENT UPDATE ###

        rx,ry,theta = self.state[0,0],self.state[1,0],self.state[2,0] # robot position (x,y) and heading 
        delta_zs = [np.zeros((2,1)) for lidx in range(self.n_landmarks)]  # Place holder for each cone (landmark)
        Ks = [np.zeros((self.state.shape[0],2)) for lidx in range(self.n_landmarks)] # A list of matrices stored for use outside the measurement for loop
        Hs = [np.zeros((2,self.state.shape[0])) for lidx in range(self.n_landmarks)] # A list of matrices stored for use outside the measurement for loop
        
        #Get all of the cones
        all_cones = np.vstack([
            yellow_cones_converted_predicted_pose_keys,
            blue_cones_converted_predicted_pose_keys,
            orange_cones_converted_predicted_pose_keys,
            orange_big_cones_converted_predicted_pose_keys
        ])

        #Get all of the indeces
        all_matched_landmarks = np.concatenate([
            matched_yellow_cones,
            matched_blue_cones,
            matched_orange_cones,
            matched_orange_big_cones
        ])

        #For each old observation
        for i,lidx in enumerate(all_matched_landmarks):

            #Skip new cones
            if lidx == -1:
                continue
            
            state_landmark = self.state[self.n_state+lidx*2:self.n_state+lidx*2+2] # Get the current estimated position of the landmark
            measured_landmark = np.array(all_cones[i]).reshape((2, 1)) # Get the measured value but with the same shape
            delta_zs[lidx] = measured_landmark - state_landmark # Difference between actual and estimated observation

            # Helper matrices in computing the measurement update
            Fxj = np.block([[self.Fx],[np.zeros((2,self.Fx.shape[1]))]])
            Fxj[self.n_state:self.n_state+2,self.n_state+2*lidx:self.n_state+2*lidx+2] = np.eye(2)
            H = Fxj  # Directly map the observed landmark components in global frame
            Hs[lidx] = H # Added to list of matrices
            Ks[lidx] = self.P.dot(np.transpose(H)).dot(np.linalg.inv(H.dot(self.P).dot(np.transpose(H)) + self.Q)) # Add to list of matrices
        
        
        # After storing appropriate matrices, perform measurement update of mu and sigma
        state_offset = np.zeros(self.state.shape) # Offset to be added to state estimate
        covariance_factor = np.eye(self.P.shape[0]) # Factor to multiply state uncertainty
        for lidx in range(self.n_landmarks):
            state_offset += Ks[lidx].dot(delta_zs[lidx]) # Compute full mu offset
            covariance_factor -= Ks[lidx].dot(Hs[lidx]) # Compute full sigma factor
        self.state = self.state + state_offset # Update state estimate
        self.P = covariance_factor.dot(self.P) # Update state uncertainty
        
        ### ADD NEW CONES ###
        self.data_augmentation(self,new_blue_cones,new_yellow_cones,new_orange_cones,new_orange_big_cones)

