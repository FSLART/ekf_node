import rclpy
from rclpy.node import Node
import numpy as np
import math
from .ekf import EKF
from lart_msgs.msg import DynamicsCMD, GNSSINS, Dynamics
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from sensor_msgs.msg import Imu, NavSatFix #from the simuator
from eufs_msgs.msg import WheelSpeedsStamped # from the simulator
from message_filters import Subscriber, ApproximateTimeSynchronizer
import time

from geometry_msgs.msg import Vector3Stamped
import matplotlib.pyplot as plt
import time

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


        ### DECLARING PARAMETERS ###

        self.declare_parameter('dynamics_cmd_topic','/acu_origin/dynamics')
        self.declare_parameter('imu_topic','/imu/angular_velocity') # TODO: this is a placeholder, change it to the correct topic

        # self.declare_parameter('dynamics_update_topic','/only/god/knows') # TODO: this is a placeholder, change it to the correct topic
        #self.declare_parameter('gnssins_topic','/ekf/state') # TODO: this is a placeholder, change it to the correct topic

        ### SUBSCRIPTIONS ###

        # Sub for Motor Speed
        dynamics_topic = self.get_parameter('dynamics_topic').get_parameter_value().string_value
        self.dynamics_sub = self.create_subscription(Dynamics, dynamics_topic, self.predict_callback, 10)

        # Sub for Imu (angular velocity)
        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        self.imu_sub = self.create_subscription(Vector3Stamped, imu_topic, self.imu_callback, 10)


        # Create message_filters subscribers
        self.imu_sub = Subscriber(self, Vector3Stamped, '/imu/angular_velocity') # IMU angular velocity
        self.speed_sub = Subscriber(self, Dynamics, '/acu_origin/dynamics') # Motor speed

        # ApproximateTimeSynchronizer (you can also use TimeSynchronizer for exact match)
        self.ts = ApproximateTimeSynchronizer(
            [self.imu_sub, self.speed_sub],
            queue_size=10,
            slop=0.5  # seconds of allowed timestamp difference
        )

        self.ts.registerCallback(self.predict_callback)

        # Create publisher
        # gnssins_topic = self.get_parameter('gnssins_topic').get_parameter_value().string_value
        # self.publisher_ = self.create_publisher(GNSSINS, gnssins_topic, 10)
        
        
        self.ekf = None

    def imu_callback(self, imu_msg):
        # Save the previous angular velocity
        self.angular_velocity = imu_msg.vector.z 
    

    def predict_callback(self, v_msg):
        if(self.ekf is None):
            self.intialize_ekf()
        
        #Convert the rpm's to m/s
        ms_speed = self.tire_perimeter * (v_msg.rpm/self.transmission_ratio/60.0)

        # Get current angular velocity from IMU
        omega_z = self.angular_velocity 

        self.get_logger().info(f"IMU: {omega_z} SPEED: {ms_speed} ")

        # Call the predict method of the EKF
        self.ekf.predict(ms_speed, omega_z)

        #plot the trajectory
        x_vals.append(float(self.ekf.state[0]))
        y_vals.append(float(self.ekf.state[1]))

        #br.set_data(y_cones, x_cones)
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

    def gns_publish(self):
        # Create a new GNSSINS message
        gnssins_msg = GNSSINS()
        gnssins_msg.position.x = self.ekf.state[0, 0]
        gnssins_msg.position.y = self.ekf.state[1, 0]
        gnssins_msg.heading = self.ekf.state[2, 0]
        # gnssins_msg.velocity.x = self.ekf.state[3, 0] * math.cos(self.ekf.state[2, 0])
        # gnssins_msg.velocity.y = self.ekf.state[3, 0] * math.sin(self.ekf.state[2, 0])
        # gnssins_msg.velocity.z = 0.0
        # Publish the GNSSINS message
        self.publisher_.publish(gnssins_msg)

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