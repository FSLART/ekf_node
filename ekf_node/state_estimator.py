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


# Read the CSV file of the track
file = open("/home/tomas/ros2_ws/src/eufs_sim/eufs_tracks/csv/small_track.csv", "r") #TODO: change to the correct file

lin = file.readlines()

x_cones = []
y_cones = []
lin.pop(0)  # Remove the first line
for l in lin:
    l = l.split(",")
    x_cones.append(float(l[1]))
    y_cones.append(float(l[2]))

file.close()


class StateEstimator(Node):

    def __init__(self):
        super().__init__('state_estimator')

        # Constants
        lart_pi = 3.14159265358979323846 
        tire_radius = 0.255  
        self.tire_perimeter = 2.0 * lart_pi * tire_radius 
        self.transmission_ratio = 4.0  
        self.previous_yaw = 0.0

        # To check the fequency
        # self.last_time = time.time()


        # # Declare parameters
        # self.declare_parameter('dynamics_cmd_topic','/pc_origin/dynamics')
        # self.declare_parameter('dynamics_update_topic','/only/god/knows') # TODO: this is a placeholder, change it to the correct topic
        #self.declare_parameter('gnssins_topic','/ekf/state') # TODO: this is a placeholder, change it to the correct topic

        # # Create subcriptions
        # dynamics_cmd_topic = self.get_parameter('dynamics_cmd_topic').get_parameter_value().string_value
        # self.dynamics_sub = self.create_subscription(DynamicsCMD, dynamics_cmd_topic, self.dynamics_callback, 100)

        # dynamics_update_topic = self.get_parameter('dynamics_update_topic').get_parameter_value().string_value
        #self._sub = self.create_subscription(Vector3Stamped, '/imu/angular_velocity', self.axanato_callback, 10)
        
        # Create message_filters subscribers
        self.imu_sub = Subscriber(self, Imu, '/imu') #imu
        #self.gps_sub = Subscriber(self, NavSatFix, '/gps')
        self.speed_sub = Subscriber(self, WheelSpeedsStamped, '/ground_truth/wheel_speeds')#/ground_truth/wheel_speeds

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

        # Define
        self.ekf = None

    '''
    def get_gnssisns(self, imu_msg, speed_msg):
        
        
        # Convert lat/lon to position in x and y
        # lat = gps_msg.latitude
        # lon = gps_msg.longitude
        # x, y = utm(lon, lat)
        # self.get_logger().info(f"GPS: {gps_msg.latitude}, {gps_msg.longitude}, LAT: {lat}, LON: {lon}")

        speed = ((speed_msg.speeds.lb_speed + speed_msg.speeds.rb_speed)/2) / 37.8188

        current_yaw = 2 * math.atan2(imu_msg.orientation.z, imu_msg.orientation.w)

        # delta = current_yaw - self.previous_yaw
        # delta = np.arctan2(np.sin(delta), np.cos(delta))

        current_yaw = (current_yaw + np.pi) % (2 * np.pi) - np.pi


        #heading = math.radians(yaw)

        #self.get_logger().info(f"IMU: {yaw}, SPEED: {speed} ")

        # Create a new GNSSINS message
        gnssins_msg = GNSSINS()
        gnssins_msg.position.x = 0.0
        gnssins_msg.position.y = 0.0
        gnssins_msg.heading = current_yaw
        gnssins_msg.velocity.x = speed
        gnssins_msg.velocity.y = 0.0
        gnssins_msg.velocity.z = 0.0

        # Call the update callback with the new message
        self.dynamics_update_callback(gnssins_msg)
    


    def dynamics_callback(self, msg):
        if(self.ekf is None):
            self.intialize_ekf()
        # Calculate the rpm to m/s
        ms_speed = msg.rpm

        # #Check the frequency
        # current_time = time.time()
        # dt = current_time - self.last_time
        # self.get_logger().info(f"Frequency: {1/dt} Hz")

        # self.last_time = current_time
        
        #print steering angle from the spac
        # steering_angle_degrees = math.degrees(msg.drive.steering_angle)
        # steering_angle_degrees = round(steering_angle_degrees, 3)
        #self.get_logger().info(f"Steering angle in degrees: {steering_angle_degrees}")

        steering_angle = msg.steering_angle
        # if steering_angle_degrees < 5:
        #     steering_angle = 0.0

        # Predict the next state
        #control_input = np.array([ms_speed, msg.steering_angle], dtype=np.float64)
        control_input = np.array([ms_speed, steering_angle], dtype=np.float64)
        self.ekf.predict(control_input)

        #write data to a file
        # file = open("ekf_log.txt", "w")
        # file.write(f"{self.ekf.state[0]},{self.ekf.state[1]},{self.ekf.state[2]}\n")

        #plot the trajectory
        x_vals.append(float(self.ekf.state[0]))
        y_vals.append(float(self.ekf.state[1]))

        br.set_data(y_cones, x_cones)
        sc.set_data(y_vals, x_vals)
        line.set_data(y_vals, x_vals)
        ax.relim()
        ax.autoscale_view()
        
        plt.draw()
        plt.pause(0.001)

        # publish the new state
        self.gns_publish()
        #self.get_logger().info(f"Predicted state: {self.ekf.state.flatten()}")
    '''

    def predict_callback(self, imu_msg, v_msg):
        if(self.ekf is None):
            self.intialize_ekf()
        
        # Get current velocity
        #v = ((v_msg.speeds.lb_speed + v_msg.speeds.rb_speed)/2) / 37.8188

        # velocidade
        v = ((v_msg.speeds.lb_speed + v_msg.speeds.rb_speed)/2) / 37.8188

        # Get current angular velocity from IMU
        omega_z = imu_msg.vector.z

        self.get_logger().info(f"IMU: {omega_z} SPEED: {v} ")

        # Call the predict method of the EKF
        self.ekf.predict(v, omega_z)

        #plot the trajectory
        x_vals.append(float(self.ekf.state[0]))
        y_vals.append(float(self.ekf.state[1]))

        br.set_data(y_cones, x_cones)
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
        initial_state = np.array([[-13.0], [10.3], [0.0], [0.0],[0.0]])  # Float dtype
        initial_covariance = np.eye(5) * (0.1**2) #changed from 4x4
        process_noise = np.diag([0.1**2, 0.1**2]).astype(np.float64)
        #process_noise = np.diag([0.1**2]).astype(np.float64)
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