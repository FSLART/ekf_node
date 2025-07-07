import rclpy
from rclpy.node import Node
import numpy as np
import math
from .ekf import EKF
from lart_msgs.msg import GNSSINS, Dynamics, ConeArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
import csv

from geometry_msgs.msg import Vector3Stamped, PoseStamped
import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
sc, = ax.plot([], [], 'ro')  # 'bo' for blue dots

bc, = ax.plot([], [], 'bo')  # blue cones
yc, = ax.plot([], [], 'yo')  # yelow cones
oc, = ax.plot([], [], 'go')  # orange cones
obc, = ax.plot([], [], 'go')  # orange big cones
ssc, = ax.plot([], [], 'ro') #selected cone 

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

        ### MOTOR SPEED VARIABLE ###
        self.angular_velocity = 0.0  # Initialize motor speed variable
        self.last_rpm = 0.0 # Initialize a safety measure for the speed


        ### DECLARING PARAMETERS ###

        self.declare_parameter('dynamics_topic','/acu_origin/dynamics')
        self.declare_parameter('imu_topic','/imu/angular_velocity')
        self.declare_parameter('cones_topic','/mapping/cones')
        self.declare_parameter('position_topic','/ekf/state')

        ### SUBSCRIPTIONS ###

        # Sub for Motor Speed
        dynamics_sub = self.get_parameter('dynamics_topic').get_parameter_value().string_value
        self.dynamics_sub = self.create_subscription(Dynamics, dynamics_sub, self.predict_callback, 10)

        # Sub for Imu (angular velocity)
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.imu_sub = self.create_subscription(Vector3Stamped, imu_topic, self.imu_callback, 10)

        # Sub for Observations
        cones_topic = self.get_parameter('cones_topic').get_parameter_value().string_value
        self.cones_sub = self.create_subscription(ConeArray, cones_topic, self.update_callback, 10)

        # Create message_filters subscribers
        self.imu_sub = Subscriber(self, Vector3Stamped, '/imu/angular_velocity') # IMU angular velocity
        self.speed_sub = Subscriber(self, Dynamics, '/acu_origin/dynamics') # Motor speed


        ### PUBLISHER ###

        # Create publisher
        position_topic = self.get_parameter('position_topic').get_parameter_value().string_value
        self.pos_pub = self.create_publisher(PoseStamped, position_topic, 10)
        
        
        self.ekf = None

    def imu_callback(self, imu_msg):
        # Save the previous angular velocity
        self.angular_velocity = imu_msg.vector.z 
    

    def predict_callback(self, v_msg):
        if self.ekf is None:
            self.intialize_ekf()

        rpm = v_msg.rpm

        # Convert the rpm's to m/s
        ms_speed = self.tire_perimeter * (rpm / self.transmission_ratio / 60.0)

        # Get current angular velocity from IMU
        omega_z = self.angular_velocity

        #self.get_logger().info(f"IMU: {omega_z} SPEED: {ms_speed}")

        # Call the predict method of the EKF
        self.ekf.predict(ms_speed, omega_z)

        # Update trajectory
        x_vals.append(float(self.ekf.state[0]))
        y_vals.append(float(self.ekf.state[1]))

        # Get cone positions from the map
        # map_blue_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.blue_cones_indices)
        # map_yellow_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.yellow_cones_indices)
        # map_orange_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_cones_indices)
        # map_orange_big_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_big_cones_indices)

        map_all_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.all_cones_indices)

        # Update cone data
        # if map_blue_cones:
        #     bc.set_data([cone[1] for cone in map_blue_cones], [cone[0] for cone in map_blue_cones])
        # if map_yellow_cones:
        #     yc.set_data([cone[1] for cone in map_yellow_cones], [cone[0] for cone in map_yellow_cones])
        # if map_orange_cones:
        #     oc.set_data([cone[1] for cone in map_orange_cones], [cone[0] for cone in map_orange_cones])
        # if map_orange_big_cones:
        #     obc.set_data([cone[1] for cone in map_orange_big_cones], [cone[0] for cone in map_orange_big_cones])

        # Update cone data
        all_maped_cones = []
        for i in range(self.ekf.n_landmarks):
            cone = (self.state[self.n_state+2*i,0], self.state[self.n_state+2*i+1,0])
            all_maped_cones.append(cone)

        bc.set_data([cone[1] for cone in all_maped_cones], [cone[0] for cone in all_maped_cones])

        # The selected cone position
        sx = self.ekf.state[3]
        sy = self.ekf.state[4]

        ssc.set_data(sy, sx)  # Update selected cone position

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

    def update_callback(self, obs_msg):
        
        if(self.ekf is None):
            self.intialize_ekf()
        
        self.ekf.update(obs_msg)

        #self.get_logger().info(f"Selected Cone: {self.ekf.state[3:5]}")

        # publish the new state
        self.position_publish()
        

    def position_publish(self):
        # Create a new PoseStamped mission
        msg = PoseStamped()
        msg.pose.position.x = self.ekf.state[0,0]
        msg.pose.position.y = self.ekf.state[1,0]
        msg.pose.orientation.w = self.ekf.state[2,0]
        self.pos_pub.publish(msg)

    def intialize_ekf(self):
        # Initialize the EKF with the initial state and covariance
        initial_state = np.array([[0.0], [0.0], [0.0]])  # Float dtype #-15 PARA SKIDPAD
        process_noise = np.diag([0.002, 0.002,0.0005]).astype(np.float64)
        wheelbase = 1.55
        self.ekf = EKF(initial_state, process_noise)

    def write_cones_to_csv(self):
        # Collect cone data
        all_maped_cones = []
        for i in range(self.ekf.n_landmarks):
            cone = (self.state[self.n_state+2*i,0], self.state[self.n_state+2*i+1,0])
            all_maped_cones.append(cone)

        # blue_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.blue_cones_indices)
        # yellow_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.yellow_cones_indices)
        # orange_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_cones_indices)
        # orange_big_cones = self.ekf.get_cones_from_map(self.ekf.state, self.ekf.orange_big_cones_indices)

        # Write to CSV
        with open('cones_coordinates.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Cone Type', 'X', 'Y'])
            for cone in all_maped_cones:
                writer.writerow(['Blue', cone[0], cone[1]])
            # for cone in blue_cones:
            #     writer.writerow(['Blue', cone[0], cone[1]])
            # for cone in yellow_cones:
            #     writer.writerow(['Yellow', cone[0], cone[1]])
            # for cone in orange_cones:
            #     writer.writerow(['Orange', cone[0], cone[1]])
            # for cone in orange_big_cones:
            #     writer.writerow(['Orange Big', cone[0], cone[1]])

    def destroy_node(self):
        # Write cones to CSV before shutting down
        if self.ekf:
            self.write_cones_to_csv()
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