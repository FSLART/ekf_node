import rclpy
from rclpy.node import Node
import numpy as np
import math
from .ekf import EKF
from lart_msgs.msg import DynamicsCMD, GNSSINS



class StateEstimator(Node):

    def __init__(self):
        super().__init__('state_estimator')

        # Declare parameters
        self.declare_parameter('dynamics_cmd_topic','pc_origin/dynamics')
        self.declare_parameter('r_dynamics_topic','/only/god/knows') # TODO: this is a placeholder, change it to the correct topic

        # Create subcriptions
        dynamics_cmd_topic = self.get_parameter('dynamics_cmd_topic').get_parameter_value().string_value
        self.dynamics_sub = self.create_subscription(DynamicsCMD, dynamics_cmd_topic, self.dynamics_callback, 10)

        r_dynamics_topic = self.get_parameter('r_dynamics_topic').get_parameter_value().string_value
        self.rpm_sub = self.create_subscription(GNSSINS, r_dynamics_topic, self.r_dynamics_callback, 10)

        # Inititalize the EKF
        initial_state = np.array([[0.0], [0.0], [0.0], [0.0]])  # Float dtype
        initial_covariance = np.eye(4) * (0.1**2)
        process_noise = np.diag([0.1**2, 0.1**2]).astype(np.float64)
        wheelbase = 1.55
        self.ekf = EKF(initial_state, initial_covariance, process_noise, wheelbase)

    def dynamics_callback(self, msg):
        # Predict the next state
        control_input = np.array([msg.rpm, msg.steering_angle], dtype=np.float64)
        self.ekf.predict(control_input)

    def r_dynamics_callback(self, msg):
        # Calculate the speed from the GNSSINS message
        # and update the EKF with the new measurement
        speed = math.sqrt(msg.velocity.x**2 + msg.velocity.y**2 + msg.velocity.z**2)
        measurement = np.array([[msg.position.x], [msg.position.y], [msg.heading], [speed]], dtype=np.float64)
        measurement_noise = np.eye(4) * 0.005
        self.ekf.update(measurement, measurement_noise)

def main(args=None):
    rclpy.init(args=args)
    state_estimator = StateEstimator()
    rclpy.spin(state_estimator)
    state_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()