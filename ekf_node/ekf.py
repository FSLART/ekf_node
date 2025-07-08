import numpy as np
from sklearn.neighbors import KDTree
import time
import rclpy.logging
from math import sqrt

class EKF(object):
    def __init__(self, initial_state, noise):
        #self.wheelbase = wheelbase

        self.logger = rclpy.logging.get_logger('ekf_logger')  # Create a logger instance
        
        #Time initialization
        self.last_time = time.time()

        # auxiliary variables
        self.n_state = 3
        self.Fx = np.eye(3)

        #Landmarks
        self.blue_cones_indices = []
        self.yellow_cones_indices = []
        self.orange_cones_indices = []
        self.orange_big_cones_indices = []
        self.n_landmarks = 0

        # Ensure initial_state is float to avoid dtype issues
        self.state = initial_state.astype(np.float64)  # [x, y, theta]
        self.P = np.zeros((self.n_state+2*self.n_landmarks,self.n_state+2*self.n_landmarks)) # Covariance matrix
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

    # def data_association(self, cones, observations, threshold):
    #     """
    #     cones         : (M, 2) array - landmarks already in the map (global frame)
    #     observations  : (N, 2) array - newly observed cones (global frame)
    #     threshold     : scalar  - max Euclidian distance [m] for a valid match

    #     Returns
    #     -------
    #     idx_map : list length N
    #             For each observation: landmark index if matched, else -1.
    #     """
    #     # Nothing to match against → everything is a new cone
    #     if len(cones) == 0:
    #         return [-1] * len(observations)

    #     # No observations of this colour in the current frame
    #     if len(observations) == 0:
    #         return []

    #     cones        = np.asarray(cones,        dtype=float)
    #     observations = np.asarray(observations, dtype=float)

    #     tree = KDTree(cones)

    #     # k=1 → nearest neighbour; dists/idx are shape (N, 1)
    #     dists, idx   = tree.query(observations, k=1)

    #     # Decide if the NN is close enough
    #     idx_map = [int(i) if d <= threshold else -1
    #             for d, i in zip(dists[:, 0], idx[:, 0])]
    #     return idx_map
    
    def data_association(self, cones, observations, obs_colors, threshold):
        '''
        Perform data association between cones and observations.
        
        Args:
            cones (list): List of cone positions (2D coordinates).
            observations (list): List of observed positions (2D coordinates).
            obs_colors (list): List of colors corresponding to each observation.
            threshold (float): Max distance for association.
        
        Returns:
            array: array the same size as observations, for each observation it returns the index of the landmark that is already in the map, -1 if its not there yet
        '''

        if len(cones) == 0:
            return [-1] * len(observations)
        
        matched_cones = []

        for idx, obs in enumerate(observations):
            # Get only the cones of the corresponding color
            color = obs_colors[idx]

            available_cones = []
            if color == 1:  # Yellow
                available_cones = self.yellow_cones_indices
            elif color == 2:  # Blue
                available_cones = self.blue_cones_indices
            elif color == 3:  # Orange
                available_cones = self.orange_cones_indices
            elif color == 4:  # Orange Big
                available_cones = self.orange_big_cones_indices

            self.logger.info(f"yellow_cones_indices: {self.yellow_cones_indices}, blue_cones_indices: {self.blue_cones_indices}, orange_cones_indices: {self.orange_cones_indices}, orange_big_cones_indices: {self.orange_big_cones_indices}")

            if len(available_cones) == 0:
                matched_cones.append(-1)
                continue

            tree = KDTree(cones)
            indices = tree.query_radius([obs], r=threshold)[0]
            if len(indices) > 0:
                for idx in indices:
                    if idx in available_cones:
                        matched_cones.append(idx)
                        break
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

                #self.logger.info(f"New state after adding landmark: {self.state}")

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
                self.Fx = np.block([[self.Fx, np.zeros((self.n_state, 2))],])

    def data_augmentation_but_with_all_cones(self, new_cones, new_cones_color):
        for i, cone_coords in enumerate(new_cones):
            landmark_x, landmark_y = cone_coords

            #Add landmark to state
            new_landmark = np.array([[landmark_x], [landmark_y]])
            self.state = np.vstack((self.state, new_landmark))

            landmark_cov = np.eye(2) * 1e3  # High initial uncertainty
            top_right = np.zeros((self.P.shape[0], 2))
            bottom_left = np.zeros((2, self.P.shape[1]))
            self.P = np.block([
                [self.P,        top_right],
                [bottom_left,   landmark_cov]
            ])

            # Save the landmarks
            if new_cones_color[i] == 1:  # Yellow
                self.yellow_cones_indices.append(self.n_landmarks)
            elif new_cones_color[i] == 2:  # Blue
                self.blue_cones_indices.append(self.n_landmarks)
            elif new_cones_color[i] == 3:  # Orange
                self.orange_cones_indices.append(self.n_landmarks)
            elif new_cones_color[i] == 4:  # Orange Big
                self.orange_big_cones_indices.append(self.n_landmarks)

            self.n_landmarks += 1
            self.Fx = np.block([[self.Fx, np.zeros((self.n_state, 2))],])

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
        
        ## TODOS OS CONES
        all_mapped_cones = []
        for i in range(self.n_landmarks):
            cone = (self.state[self.n_state+2*i,0], self.state[self.n_state+2*i+1,0])
            all_mapped_cones.append(cone)

        ## DIVIDIDOS POR COR
        # map_blue_cones = self.get_cones_from_map(self.state, self.blue_cones_indices)
        # map_yellow_cones = self.get_cones_from_map(self.state, self.yellow_cones_indices)
        # map_orange_cones = self.get_cones_from_map(self.state, self.orange_cones_indices)
        # map_orange_big_cones = self.get_cones_from_map(self.state, self.orange_big_cones_indices)
        #self.logger.info(f"Step 1 - Map Cones: Blue: {len(map_blue_cones)}, Yellow: {len(map_yellow_cones)}, Orange: {len(map_orange_cones)}, Orange Big: {len(map_orange_big_cones)}")


        #separates the cones in the map by color
        blue_cones_converted_predicted_pose = {}
        yellow_cones_converted_predicted_pose = {}
        orange_cones_converted_predicted_pose = {}
        orange_big_cones_converted_predicted_pose = {}
        all_cones_converted_predicted_pose = {}
        pose_x, pose_y, pose_theta = self.state[0], self.state[1], self.state[2]

        for i,obs in enumerate(z.cones):
            obs_x, obs_y = obs.position.x, obs.position.y
            color = obs.class_type
            # Calculate the expected observation 
            if (sqrt(obs_x**2 + obs_y**2) > 10.0 or abs(obs_y) > 4.0):
                continue
            expected_obs_x = pose_x + np.cos(pose_theta) * obs_x - np.sin(pose_theta) * obs_y
            expected_obs_y = pose_y + np.sin(pose_theta) * obs_x + np.cos(pose_theta) * obs_y

            #TUDO AO MOLHO E FE EM DEUS
            all_cones_converted_predicted_pose[(float(expected_obs_x), float(expected_obs_y))] = color.data

            # CONES DIVIDIDOS POR COR

            # if color.data == 1:
            #     yellow_cones_converted_predicted_pose[(float(expected_obs_x), float(expected_obs_y))] = i
            # elif color.data == 2:
            #     blue_cones_converted_predicted_pose[(float(expected_obs_x), float(expected_obs_y))] = i
            # elif color.data == 3:
            #     orange_cones_converted_predicted_pose[(float(expected_obs_x), float(expected_obs_y))] = i
            # elif color.data == 4:
            #     orange_big_cones_converted_predicted_pose[(float(expected_obs_x), float(expected_obs_y))] = i

        #gets the global position of the cones
        
        ##TODOS OS CONES BANGERRRR
        all_cones_converted_predicted_pose_keys = np.array(list(all_cones_converted_predicted_pose.keys()))
        all_cones_converted_predicted_pose_colors = np.array(list(all_cones_converted_predicted_pose.values()))

        ##  DIVIDIDOS POR COR
        # yellow_cones_converted_predicted_pose_keys = np.array(list(yellow_cones_converted_predicted_pose.keys()))
        # blue_cones_converted_predicted_pose_keys = np.array(list(blue_cones_converted_predicted_pose.keys()))
        # orange_cones_converted_predicted_pose_keys = np.array(list(orange_cones_converted_predicted_pose.keys()))
        # orange_big_cones_converted_predicted_pose_keys = np.array(list(orange_big_cones_converted_predicted_pose.keys()))
        

        #perform data association

        ## TODOS OS CONES
        matched_all_cones = self.data_association(all_mapped_cones, all_cones_converted_predicted_pose_keys, all_cones_converted_predicted_pose_colors, 2.2)
        #self.logger.info(f"Step 2 - matched Cones: {matched_all_cones}")
        ## DIVIDIDOS POR COR

        # matched_yellow_cones = self.data_association(map_yellow_cones, yellow_cones_converted_predicted_pose_keys, 2.0)
        # matched_blue_cones = self.data_association(map_blue_cones, blue_cones_converted_predicted_pose_keys, 2.0)
        # matched_orange_cones = self.data_association(map_orange_cones, orange_cones_converted_predicted_pose_keys, 2.0)
        # matched_orange_big_cones = self.data_association(map_orange_big_cones, orange_big_cones_converted_predicted_pose_keys, 2.0)

        ## TODOS OS CONES
        all_new_cones = []
        all_new_cones_color = []
        for i in range(len(matched_all_cones)):
            if matched_all_cones[i] == -1:
                all_new_cones.append(tuple(all_cones_converted_predicted_pose_keys[i]))
                all_new_cones_color.append(all_cones_converted_predicted_pose_colors[i])
            # else:
            #     self.logger.info(f"Matched cone at {tuple(cords)} with index {matched_all_cones[i]}")

        # self.logger.info(f"Step 2 - received Cones: {len(z.cones)} new cones detected")
        ## DIVIDIDOS POR COR

        # new_yellow_cones = []
        # new_blue_cones = []
        # new_orange_cones = []
        # new_orange_big_cones = []


        # for i, cords in enumerate(yellow_cones_converted_predicted_pose_keys):
        #     if matched_yellow_cones[i] == -1:
        #         new_yellow_cones.append(tuple(cords))

        # for i, cords in enumerate(blue_cones_converted_predicted_pose_keys):
        #     if matched_blue_cones[i] == -1:
        #         new_blue_cones.append(tuple(cords))

        # for i, cords in enumerate(orange_cones_converted_predicted_pose_keys):
        #     if matched_orange_cones[i] == -1:
        #         new_orange_cones.append(tuple(cords))

        # for i, cords in enumerate(orange_big_cones_converted_predicted_pose_keys):
        #     if matched_orange_big_cones[i] == -1:
        #         new_orange_big_cones.append(tuple(cords))


        ### MEASUREMENT UPDATE ###

        rx,ry,theta = self.state[0,0],self.state[1,0],self.state[2,0] # robot position (x,y) and heading 
        delta_zs = [np.zeros((2,1)) for lidx in range(self.n_landmarks)]  # Place holder for each cone (landmark)
        Ks = [np.zeros((self.state.shape[0],2)) for lidx in range(self.n_landmarks)] # A list of matrices stored for use outside the measurement for loop
        Hs = [np.zeros((2,self.state.shape[0])) for lidx in range(self.n_landmarks)] # A list of matrices stored for use outside the measurement for loop
        
        #Only chose non empty cone arrays
        # arrays_to_concat = [
        #     np.array(arr) for arr in [
        #         yellow_cones_converted_predicted_pose_keys,
        #         blue_cones_converted_predicted_pose_keys,
        #         orange_cones_converted_predicted_pose_keys,
        #         orange_big_cones_converted_predicted_pose_keys
        #     ] if len(arr) > 0
        # ]  

        #Get all of the cones
        #all_cones = np.concatenate(arrays_to_concat)
        all_cones = all_cones_converted_predicted_pose_keys

        #Only Get non empty match arrays
        # matchs_to_contat = [
        #     np.array(arr) for arr in [
        #         matched_yellow_cones,
        #         matched_blue_cones,
        #         matched_orange_cones,
        #         matched_orange_big_cones
        #     ] if len(arr) > 0
        # ]

        #Get all of the indeces
        #all_matched_landmarks = np.concatenate(matchs_to_contat)
        all_matched_landmarks = matched_all_cones


        #For each old observation
        for i,lidx in enumerate(all_matched_landmarks):
            #self.logger.info(f"Updating with landmark {i} at index {lidx} with coordinates {all_cones[i]}")
            #Skip new cones
            if lidx == -1:
                continue
            
            state_landmark = self.state[self.n_state+lidx*2:self.n_state+lidx*2+2] # Get the current estimated position of the landmark

            # if lidx == 0:
            #     self.logger.info(f"Updating with landmark {i} at index {lidx} and state_landmark {state_landmark}")

            measured_landmark = np.array(all_cones[i]).reshape((2, 1)) # Get the measured value but with the same shape
            delta_zs[lidx] = measured_landmark - state_landmark # Difference between actual and estimated observation
            H = np.zeros((2, self.state.shape[0]))
            H[:, self.n_state + 2*lidx : self.n_state + 2*lidx + 2] = np.eye(2)
            Hs[lidx] = H

            # Kalman Gain
            S = H @ self.P @ H.T + self.Q
            K = self.P @ H.T @ np.linalg.inv(S)
            Ks[lidx] = K
        

        # After storing appropriate matrices, perform measurement update of mu and sigma
        state_offset = np.zeros(self.state.shape) # Offset to be added to state estimate
        covariance_factor = np.eye(self.P.shape[0]) # Factor to multiply state uncertainty

        init_time = time.time()

        for lidx in range(self.n_landmarks):
            state_offset += Ks[lidx].dot(delta_zs[lidx]) # Compute full mu offset
            covariance_factor -= Ks[lidx].dot(Hs[lidx]) # Compute full sigma factor

        self.state = self.state + state_offset # Update state estimate
        self.P = covariance_factor.dot(self.P) # Update state uncertainty

        final_time = time.time()
        dt = final_time - init_time
        #self.logger.info(f"Measurement update took {dt:.4f} seconds")

        ### ADD NEW CONES ###
        #self.data_augmentation(new_blue_cones,new_yellow_cones,new_orange_cones,new_orange_big_cones)
        self.data_augmentation_but_with_all_cones(all_new_cones, all_new_cones_color)

