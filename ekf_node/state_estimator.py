import rclpy
from rclpy.node import Node
import numpy as np
import math
from .ekf import EKF
from lart_msgs.msg import GNSSINS, Dynamics, ConeArray
from message_filters import Subscriber, ApproximateTimeSynchronizer

from geometry_msgs.msg import Vector3Stamped, PoseStamped
import matplotlib.pyplot as plt

plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
sc, = ax.plot([], [], 'bo')  # 'bo' for blue dots
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
        if(self.ekf is None):
            self.intialize_ekf()

        rpm = v_msg.rpm

        #Convert the rpm's to m/s
        ms_speed = self.tire_perimeter * (rpm/self.transmission_ratio/60.0)

        # Get current angular velocity from IMU
        omega_z = self.angular_velocity 

        self.get_logger().info(f"IMU: {omega_z} SPEED: {ms_speed} ")

        # Call the predict method of the EKF
        self.ekf.predict(ms_speed, omega_z)

        #plot the trajectory
        x_vals.append(float(self.ekf.state[0]))
        y_vals.append(float(self.ekf.state[1]))


        # Extract landmark coordinates from EKF state
        landmarks = self.ekf.state[3:].reshape(-1, 2)  # skip x, y, theta, then reshape

        # Separate landmarks by color using index lists
        blue_cones = [landmarks[i] for i in self.ekf.blue_cones_indices]
        yellow_cones = [landmarks[i] for i in self.ekf.yellow_cones_indices]
        orange_cones = [landmarks[i] for i in self.ekf.orange_cones_indices]
        orange_big_cones = [landmarks[i] for i in self.ekf.orange_big_cones_indices]

        #br.set_data(y_cones, x_cones)
        sc.set_data(y_vals, x_vals)
        line.set_data(y_vals, x_vals)
        ax.relim()
        ax.autoscale_view()
        
        plt.draw()
        plt.pause(0.001)

        #Publish the new state
        self.position_publish()

    def update_callback(self, obs_msg):
        
        if(self.ekf is None):
            self.intialize_ekf()
        self.ekf.update(obs_msg)

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
        initial_state = np.array([[-15.0], [0.0], [0.0]])  # Float dtype
        process_noise = np.diag([0.002, 0.002,0.0005]).astype(np.float64)
        wheelbase = 1.55
        self.ekf = EKF(initial_state, process_noise)


def main(args=None):
    rclpy.init(args=args)
    state_estimator = StateEstimator()
    rclpy.spin(state_estimator)
    state_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()