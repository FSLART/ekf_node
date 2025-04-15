import rclpy
from rclpy.node import Node

from std_msgs.msg import String


class StateEstimator(Node):

    def __init__(self):
        super().__init__('state_estimator')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)
    state_estimator = StateEstimator()
    rclpy.spin(state_estimator)
    state_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()