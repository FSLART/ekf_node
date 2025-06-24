import rclpy
from rclpy.node import Node
import numpy as np
import math
from .ekf import EKF
from lart_msgs.msg import DynamicsCMD, GNSSINS
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from sensor_msgs.msg import Imu, NavSatFix #from the simuator
from eufs_msgs.msg import WheelSpeedsStamped # from the simulator
from message_filters import Subscriber, ApproximateTimeSynchronizer
import time

from geometry_msgs.msg import Vector3Stamped, PoseStamped
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
        self.previous_yaw = 0.0

        ### MOTOR SPEED VARIABLE ###
        self.angular_velocity = 0.0  # Initialize motor speed variable

        # Create subcriptions
        self.speed_sub = self.create_subscription(WheelSpeedsStamped, '/ground_truth/wheel_speeds', self.predict_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        
        # Create publisher
        self.pos_pub = self.create_publisher(PoseStamped, "/ekf/state", 10)
        
        
        self.ekf = None


    def imu_callback(self, imu_msg):
        # Save the previous angular velocity
        self.angular_velocity = imu_msg.angular_velocity.z


    def predict_callback(self, v_msg):
        if(self.ekf is None):
            self.intialize_ekf()
        
        #Get current velocity
        v = ((v_msg.speeds.lb_speed + v_msg.speeds.rb_speed)/2) / 37.8188

        # Get current angular velocity from IMU
        omega_z = self.angular_velocity

        self.get_logger().info(f"IMU: {omega_z} SPEED: {v} ")

        # Call the predict method of the EKF
        self.ekf.predict(v, omega_z)

        #plot the trajectory
        x_vals.append(float(self.ekf.state[0]))
        y_vals.append(float(self.ekf.state[1]))

        sc.set_data(y_vals, x_vals)
        line.set_data(y_vals, x_vals)
        ax.relim()
        ax.autoscale_view()
        
        plt.draw()
        plt.pause(0.001)


    def dynamics_update_callback(self, msg):
        if(self.ekf is None):
            self.intialize_ekf()
        # Calculate the speed from the GNSSINS message
        # and update the EKF with the new measurement
        # speed = math.sqrt(msg.velocity.x**2 + msg.velocity.y**2)

        #Check the frequency
        # current_time = time.time()
        # dt = current_time - self.last_time
        # self.get_logger().info(f"Frequency: {1/dt} Hz")

        # self.last_time = current_time
        
        speed = msg.velocity.x # AXANATO

        self.get_logger().info(f"Predicted state: {self.ekf.state[2,0]}")
        self.get_logger().info(f"SIMULADOR: {msg.heading}")

        measurement = np.array([[self.ekf.state[0,0]], [self.ekf.state[1,0]], [msg.heading]], dtype=np.float64) #REMOVIDA A SPEED
        measurement_noise = np.eye(3) * 0.005 # 4
        self.ekf.update(measurement, measurement_noise)
        # publish the new state
        self.gns_publish()

    def position_publish(self):
        # Create a new PoseStamped mission
        msg = PoseStamped()
        msg.pose.position.x = self.ekf.state[0,0]
        msg.pose.position.y = self.ekf.state[1,0]
        msg.pose.orientation.w = self.ekf.state[2,0]
        self.pos_pub.publish(msg)

    def intialize_ekf(self):
        # Initialize the EKF with the initial state and covariance
        #initial_state = np.array([[-13.0], [10.3], [0.0]])  # Float dtype
        initial_state = np.array([[0.0], [0.0], [0.0]])  # Float dtype

        process_noise = np.diag([0.002, 0.002,0.0005]).astype(np.float64)
        #process_noise = np.diag([0.1**2]).astype(np.float64)
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