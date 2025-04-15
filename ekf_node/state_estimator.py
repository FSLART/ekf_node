import rclpy
from rclpy.node import Node
from .ekf import EKF
from lart_msgs.msg import DynamicsCMD, Dynamics

from std_msgs.msg import String


class StateEstimator(Node):

    def __init__(self):
        super().__init__('state_estimator')

        self.declare_parameter('dynamics_cmd_topic','pc_origin/dynamics')
        self.declare_parameter('rpm_topic','/acu_origin/dynamics')
        self.declare_parameter('steering_topic','/only/god/knows') # TODO: this is a placeholder, change it to the correct topic

        dynamics_cmd_topic = self.get_parameter('dynamics_cmd_topic').get_parameter_value().string_value
        self.dynamics_sub = self.create_subscription(DynamicsCMD, dynamics_cmd_topic, self.dynamics_callback, 10)

        rpm_topic = self.get_parameter('rpm_topic').get_parameter_value().string_value
        self.rpm_sub = self.create_subscription(Dynamics, rpm_topic, self.rpm_callback, 10)

        steering_topic = self.get_parameter('steering_topic').get_parameter_value().string_value
        self.steering_sub = self.create_subscription(float, steering_topic, self.steering_callback, 10)

        self.ekf = EKF()

    def dynamics_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

    def rpm_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
    
    def steering_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)
    state_estimator = StateEstimator()
    rclpy.spin(state_estimator)
    state_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()