import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
import numpy as np
from .ekf import EKF
from lart_msgs.msg import Dynamics, ConeArray, Cone, SlamStats, Mission
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import UInt16
import csv

from geometry_msgs.msg import Vector3Stamped, PoseStamped
import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
sc, = ax.plot([], [], 'ro')  # 'bo' for blue dots

bc, = ax.plot([], [], 'o', color='#203cd6')  # blue cones
yc, = ax.plot([], [], 'o', color='#e2b80a')  # yelow cones
oc, = ax.plot([], [], 'o', color='#fea92e')  # orange cones
obc, = ax.plot([], [], 'o', color='#f25b05')  # orange big cones
ssc, = ax.plot([], [], 'o', color='#f254f0') #selected cone 

line, = ax.plot([], [], 'b-')  # line to show trajectory
br, = ax.plot([], [], 'ro')  # red dots for cones
ax.set_xlim(-10, 10)  # You can adjust these as needed
ax.set_ylim(-10, 10)
ax.set_xlabel("State[1] (y)")
ax.set_ylabel("State[0] (x)")
ax.set_title("EKF Trajectory")
ax.xaxis.set_inverted(True)  # Invert x-axis
ax.grid(True)
ax.axis('equal')

x_vals = []
y_vals = []

class StateEstimator(Node):

    def __init__(self):
        super().__init__('state_estimator')

        ### COSTANTS ###

        lart_pi = 3.14159265358979323846 
        tire_radius = 0.255  
        self.tire_perimeter = 2.0 * lart_pi * tire_radius 
        self.transmission_ratio = 4.0  
        self.previous_yaw = 0.0
        self.min_lap_dist = 15.0

        ### LAP COUNTER VARIABLES ###
        self.lap_count = -1 # Laps

        self.margin_x = 1.0
        self.margin_y = 3.0

        ### MOTOR SPEED VARIABLE ###
        self.angular_velocity = 0.0  # Initialize motor speed variable
        self.last_rpm = 0.0 # Initialize a safety measure for the speed

        ### MISSION VARIABLES ###
        self.mission = Mission.MANUAL #Consider Manual a the default mission


        ### DECLARING PARAMETERS ###

        self.declare_parameter('dynamics_topic','/acu_origin/dynamics')
        self.declare_parameter('imu_topic','/imu/angular_velocity')
        self.declare_parameter('cones_topic','/mapping/cones')
        self.declare_parameter('position_topic','/ekf/state')
        self.declare_parameter('map_topic','/ekf/map')
        self.declare_parameter('markers_topic','/ekf/cone_markers')
        self.declare_parameter('stats_topic', '/ekf/stats')
        self.declare_parameter('mission_topic','/pc_origin/system_status/critical_as/mission')

        ### SUBSCRIPTIONS ###

        # Sub for Motor Speed
        dynamics_topic = self.get_parameter('dynamics_topic').get_parameter_value().string_value
        self.dynamics_sub = self.create_subscription(Dynamics, dynamics_topic, self.predict_callback, 10)

        # Sub for Imu (angular velocity)
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.imu_sub = self.create_subscription(Vector3Stamped, imu_topic, self.imu_callback, 10)

        # Sub for Observations
        cones_topic = self.get_parameter('cones_topic').get_parameter_value().string_value
        self.cones_sub = self.create_subscription(ConeArray, cones_topic, self.update_callback, 10)

        # Sub for mission
        mission_topic = self.get_parameter('mission_topic').get_parameter_value().string_value
        self.mission_sub = self.get_parameter(Mission,mission_topic, self.mission_callback, 10)


        ### PUBLISHER ###

        # Create publisher
        position_topic = self.get_parameter('position_topic').get_parameter_value().string_value
        self.pos_pub = self.create_publisher(PoseStamped, position_topic, 10)

        # Map publisher
        map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.map_pub = self.create_publisher(ConeArray, map_topic, 10)

        # Cone markers publisher
        markers_topic = self.get_parameter('markers_topic').get_parameter_value().string_value
        self.markers_pub = self.create_publisher(MarkerArray, markers_topic, 10)

        # Slam stats publisher
        stats_topic = self.get_parameter('stats_topic').get_parameter_value().string_value
        self.slam_stats_pub = self.create_publisher(SlamStats, stats_topic, 10)

        
        ### AUX VARIABLES ###
        self.last_rpm = 0.0  # Initialize last rpm to zero
        self.count_marker = 0
        
        self.ekf = None

        self.map_timer = self.create_timer(0.02, self.map_publish)  # Timer to publish map at 50Hz
        self.slam_stats_timer = self.create_timer(0.02, self.publish_slam_stats)  # Timer to publish slam stats at 50Hz

    def mission_callback(self, msg):
        self.mission = msg.data


    def lap_callback(self, msg):
        if self.ekf is None:
            return
        
        self.lap_count = msg.data

        ### Post Processing ###
        if self.ekf.lap_count == 1:
            self.ekf.post_processing()  # Process the cones after the first lap

    def imu_callback(self, imu_msg):
        # Save the previous angular velocity
        self.angular_velocity = imu_msg.vector.z 
    

    def predict_callback(self, v_msg):
        if self.ekf is None:
            self.intialize_ekf()
            return

        rpm = 0

        if v_msg.rpm > 2000:
            rpm = self.last_rpm  # If the rpm is too high, use the last known rpm
        else:
            self.last_rpm = rpm
            rpm = v_msg.rpm

        # Convert the rpm's to m/s
        ms_speed = self.tire_perimeter * (rpm / self.transmission_ratio / 60.0)

        # Get current angular velocity from IMU
        omega_z = self.angular_velocity

        # Call the predict method of the EKF
        self.ekf.predict(ms_speed, omega_z)

        
        # Update trajectory
        x_vals.append(float(self.ekf.state[0]))
        y_vals.append(float(self.ekf.state[1]))

        # Get cone positions from the map
        map_blue_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.blue_cones_indices)
        map_yellow_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.yellow_cones_indices)
        map_orange_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_cones_indices)
        map_orange_big_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_big_cones_indices)

        # Update cone data
        if map_blue_cones:
            bc.set_data([cone[1] for cone in map_blue_cones], [cone[0] for cone in map_blue_cones])
        if map_yellow_cones:
            yc.set_data([cone[1] for cone in map_yellow_cones], [cone[0] for cone in map_yellow_cones])
        if map_orange_cones:
            oc.set_data([cone[1] for cone in map_orange_cones], [cone[0] for cone in map_orange_cones])
        if map_orange_big_cones:
            obc.set_data([cone[1] for cone in map_orange_big_cones], [cone[0] for cone in map_orange_big_cones])

        # Update trajectory plot
        sc.set_data(y_vals, x_vals)
        line.set_data(y_vals, x_vals)

        # Refresh plot
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)
        

        # Publish the new state
        self.position_publish()

        # Verify if a lap was completed
        self.verify_lap()

    def update_callback(self, obs_msg):
        if self.ekf is None:
            self.intialize_ekf()
            return
        
        self.ekf.update(obs_msg)

        # publish the new state
        self.position_publish()

        # Verify if a lap was completed
        self.verify_lap()
        

    def position_publish(self):
        # Create a new PoseStamped mission
        msg = PoseStamped()
        msg.pose.position.x = self.ekf.state[0,0]
        msg.pose.position.y = self.ekf.state[1,0]
        msg.pose.orientation.w = self.ekf.state[2,0]
        self.pos_pub.publish(msg)
    
    def map_publish(self):
        if self.ekf is None:
            return
        
        # Intialize ConeArray msg
        cone_array_msg = ConeArray()

        marker_array_msg = MarkerArray()

        # Get cone by color
        map_yellow_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.yellow_cones_indices)
        for cone in map_yellow_cones:
            cone_aux = Cone()
            cone_aux.position.x = cone[0]
            cone_aux.position.y = cone[1]
            cone_aux.class_type.data = 1 # Yellow cone
            cone_array_msg.cones.append(cone_aux)
            '''
            marker = self.create_marker(cone, 1) 
            marker_array_msg.markers.append(marker)
            '''


        map_blue_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.blue_cones_indices)
        for cone in map_blue_cones:
            cone_aux = Cone()
            cone_aux.position.x = cone[0]
            cone_aux.position.y = cone[1]
            cone_aux.class_type.data = 2 # Blue cone
            cone_array_msg.cones.append(cone_aux)
            '''
            marker = self.create_marker(cone, 2)  # Create marker for blue cone
            marker_array_msg.markers.append(marker)
            '''
        
        map_orange_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_cones_indices)
        for cone in map_orange_cones:
            cone_aux = Cone()
            cone_aux.position.x = cone[0]
            cone_aux.position.y = cone[1]
            cone_aux.class_type.data = 3 # Orange cone
            cone_array_msg.cones.append(cone_aux)
            '''
            marker = self.create_marker(cone, 3) 
            marker_array_msg.markers.append(marker)
            '''

        map_orange_big_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_big_cones_indices)
        for cone in map_orange_big_cones:
            cone_aux = Cone()
            cone_aux.position.x = cone[0]
            cone_aux.position.y = cone[1]
            cone_aux.class_type.data = 4 # Orange Big cone
            cone_array_msg.cones.append(cone_aux)
            '''
            marker = self.create_marker(cone, 4) 
            marker_array_msg.markers.append(marker)
            '''

        # Publish the ConeArray message
        self.map_pub.publish(cone_array_msg)
        '''
        # Publish the MarkerArray message
        self.markers_pub.publish(marker_array_msg)
        '''

    def publish_slam_stats(self):
        if self.ekf is None:
            return
        #Create a new SlamStats message
        slam_stats_msg = SlamStats()
        slam_stats_msg.lap_count = self.lap_count
        slam_stats_msg.cones_count_all = self.ekf.n_landmarks
        slam_stats_msg.cones_count_current = self.ekf.current_n_observations

        self.slam_stats_pub.publish(slam_stats_msg)



    def verify_lap(self):

        position_x = self.ekf.state[0,0]
        position_y = self.ekf.state[1,0]

        if self.lap_count == -1:
            if self.mission == Mission.SKIDPAD:
                if np.abs(position_x - 15.0) < self.margin_x:
                    #Only start counting in the midle of the skidpad
                    self.lap_count = 0
                    distance_after_lap = self.ekf.distance
                    return
                else:
                    #Initialize other missions with lap 0
                    self.lap_count = 0
                    distance_after_lap = self.ekf.distance

        # Prevent lap from incrementing multiple times in the same real lap
        dt_dist = self.ekf.distance - distance_after_lap #TODO: CHECK THIS
        if dt_dist < self.min_lap_dist:
            return

        lap_complete = False

        #Aceleration Lap
        if self.mission == Mission.ACCELERATION:
            if np.abs(position_x - 75.0) < self.margin_x:
                lap_complete = True
        #SkidPad lap
        if self.mission == Mission.SKIDPAD:
            if (np.abs(position_x - 15) < self.margin_x) and (np.abs(position_y) < self.margin_y):
                lap_complete = True
        #TrackDrive and Autocross lap
        if self.mission == Mission.AUTOCROSS or self.mission == Mission.TRACKDRIVE:
            if np.abs(position_x) < self.margin_x and np.abs(position_y) < self.margin_y:
                lap_complete = True
        
        #Increment Lap
        if lap_complete:
            self.lap_count += 1      
            distance_after_lap = self.ekf.distance


    def create_marker(self, cone, cone_type):
        marker = Marker()
        marker.header.frame_id = 'base_footprint'
        marker.id = self.count_marker
        self.count_marker += 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.lifetime = Duration(seconds=0.1).to_msg()
        marker.pose.position.x = cone[0]
        marker.pose.position.y = cone[1]
        marker.pose.position.z = 0.0

        if cone_type == 1:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.25
        elif cone_type == 2:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.25
        elif cone_type == 3:
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.25
        elif cone_type == 4:
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.35
            marker.scale.y = 0.35
            marker.scale.z = 0.50

        return marker
            


    def intialize_ekf(self):
        # Initialize the EKF with the initial state and covariance
        if self.mission == Mission.MANUAL:
            return
        elif self.mission == Mission.TRACKDRIVE or self.mission == Mission.AUTOCROSS:
            initial_state = np.array([[-6.0], [0.0], [0.0]])
        else:
            initial_state = np.array([[0.0], [0.0], [0.0]])
        process_noise = np.diag([0.002, 0.002,0.0005]).astype(np.float64)
        self.ekf = EKF(initial_state, process_noise)


    def write_cones_to_csv(self):
        # Collect cone data
        blue_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.blue_cones_indices)
        yellow_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.yellow_cones_indices)
        orange_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_cones_indices)
        orange_big_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_big_cones_indices)

        # Write to CSV
        with open('cones_coordinates.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Cone Type', 'X', 'Y', 'covariance'])
            for i,cone in enumerate(blue_cones):
                writer.writerow(['Blue', cone[0], cone[1], cone[2]]) 
            for cone in yellow_cones:
                writer.writerow(['Yellow', cone[0], cone[1], cone[2]])
            for cone in orange_cones:
                writer.writerow(['Orange', cone[0], cone[1], cone[2]])
            for cone in orange_big_cones:
                writer.writerow(['Orange Big', cone[0], cone[1], cone[2]])
            

    def destroy_node(self):
        # Write cones to CSV before shutting down
        if self.ekf:
            self.ekf.post_processing()
            self.write_cones_to_csv()

            # Get cone positions from the map
            map_blue_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.blue_cones_indices)
            map_yellow_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.yellow_cones_indices)
            map_orange_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_cones_indices)
            map_orange_big_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_big_cones_indices)

            # Update cone data
            if map_blue_cones:
                bc.set_data([cone[1] for cone in map_blue_cones], [cone[0] for cone in map_blue_cones])
            if map_yellow_cones:
                yc.set_data([cone[1] for cone in map_yellow_cones], [cone[0] for cone in map_yellow_cones])
            if map_orange_cones:
                oc.set_data([cone[1] for cone in map_orange_cones], [cone[0] for cone in map_orange_cones])
            if map_orange_big_cones:
                obc.set_data([cone[1] for cone in map_orange_big_cones], [cone[0] for cone in map_orange_big_cones])

            # Update trajectory plot
            sc.set_data(y_vals, x_vals)
            line.set_data(y_vals, x_vals)

            # Refresh plot
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(60.0)

        super().destroy_node()


def main(args=None):
    try:
        rclpy.init(args=args)
        state_estimator = StateEstimator()
        rclpy.spin(state_estimator)
    except KeyboardInterrupt:
        state_estimator.get_logger().info('State Estimator Node terminated.')
    finally:    
        state_estimator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()