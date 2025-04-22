import rclpy
from rclpy.node import Node
import numpy as np
import math
from .ekf import EKF
from lart_msgs.msg import DynamicsCMD, GNSSINS

import matplotlib.pyplot as plt
import time

plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
sc, = ax.plot([], [], 'bo')  # 'bo' for blue dots
line, = ax.plot([], [], 'b-')  # line to show trajectory
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

        # Constants
        lart_pi = 3.14159265358979323846 
        tire_radius = 0.255  
        self.tire_perimeter = 2.0 * lart_pi * tire_radius 
        self.transmission_ratio = 4.0  

        # Declare parameters
        self.declare_parameter('dynamics_cmd_topic','pc_origin/dynamics')
        self.declare_parameter('dynamics_update_topic','/only/god/knows') # TODO: this is a placeholder, change it to the correct topic
        self.declare_parameter('gnssins_topic','/only/god/knows/two') # TODO: this is a placeholder, change it to the correct topic

        # Create subcriptions
        dynamics_cmd_topic = self.get_parameter('dynamics_cmd_topic').get_parameter_value().string_value
        self.dynamics_sub = self.create_subscription(DynamicsCMD, dynamics_cmd_topic, self.dynamics_callback, 10)

        dynamics_update_topic = self.get_parameter('dynamics_update_topic').get_parameter_value().string_value
        self.rpm_sub = self.create_subscription(GNSSINS, dynamics_update_topic, self.dynamics_update_callback, 10)

        # Create publisher
        gnssins_topic = self.get_parameter('gnssins_topic').get_parameter_value().string_value
        self.publisher_ = self.create_publisher(GNSSINS, gnssins_topic, 10)

        # Define
        self.ekf = None

    def dynamics_callback(self, msg):
        if(self.ekf is None):
            self.intialize_ekf()
        # Calculate the rpm to m/s
        ms_speed = self.tire_perimeter * (msg.rpm/self.transmission_ratio/60.0)
        
        #print steering angle from the spac
        steering_angle_degrees = math.degrees(msg.steering_angle)
        steering_angle_degrees = round(steering_angle_degrees, 3)
        self.get_logger().info(f"Steering angle in degrees: {steering_angle_degrees}")

        steering_angle = msg.steering_angle
        # if steering_angle_degrees < 5:
        #     steering_angle = 0.0

        # Predict the next state
        #control_input = np.array([ms_speed, msg.steering_angle], dtype=np.float64)
        control_input = np.array([ms_speed, steering_angle], dtype=np.float64)
        self.ekf.predict(control_input)

        #write data to a file
        file = open("ekf_log.txt", "w")
        file.write(f"{self.ekf.state[0]},{self.ekf.state[1]},{self.ekf.state[2]}\n")

        #plot the trajectory
        x_vals.append(float(self.ekf.state[0]))
        y_vals.append(float(-self.ekf.state[1]))

        sc.set_data(y_vals, x_vals)
        line.set_data(y_vals, x_vals)
        ax.relim()
        ax.autoscale_view()
        
        plt.draw()
        plt.pause(0.1)

        # publish the new state
        self.gns_publish()
        self.get_logger().info(f"Predicted state: {self.ekf.state.flatten()}")

    def dynamics_update_callback(self, msg):
        if(self.ekf is None):
            self.intialize_ekf()
        # Calculate the speed from the GNSSINS message
        # and update the EKF with the new measurement
        speed = math.sqrt(msg.velocity.x**2 + msg.velocity.y**2)
        measurement = np.array([[msg.position.x], [msg.position.y], [msg.heading], [speed]], dtype=np.float64)
        measurement_noise = np.eye(4) * 0.005
        self.ekf.update(measurement, measurement_noise)
        # publish the new state
        self.gns_publish()

    def gns_publish(self):
        # Create a new GNSSINS message
        gnssins_msg = GNSSINS()
        gnssins_msg.position.x = self.ekf.state[0, 0]
        gnssins_msg.position.y = self.ekf.state[1, 0]
        gnssins_msg.heading = self.ekf.state[2, 0]
        gnssins_msg.velocity.x = self.ekf.state[3, 0] * math.cos(self.ekf.state[2, 0])
        gnssins_msg.velocity.y = self.ekf.state[3, 0] * math.sin(self.ekf.state[2, 0])
        gnssins_msg.velocity.z = 0.0
        # Publish the GNSSINS message
        self.publisher_.publish(gnssins_msg)

    def intialize_ekf(self):
        # Initialize the EKF with the initial state and covariance
        initial_state = np.array([[0.0], [0.0], [0.0], [0.0]])  # Float dtype
        initial_covariance = np.eye(4) * (0.1**2)
        process_noise = np.diag([0.1**2, 0.1**2]).astype(np.float64)
        wheelbase = 1.55
        self.ekf = EKF(initial_state, initial_covariance, process_noise, wheelbase)

def main(args=None):
    rclpy.init(args=args)
    state_estimator = StateEstimator()
    rclpy.spin(state_estimator)
    state_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()