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
        # publish the new state
        self.gns_publish()
        self.get_logger().info(f"Predicted state: {self.ekf.state.flatten()}")

    def dynamics_update_callback(self, msg):
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


def main(args=None):
    rclpy.init(args=args)
    state_estimator = StateEstimator()
    rclpy.spin(state_estimator)
    state_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()