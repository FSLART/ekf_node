from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ekf_node',
            executable='ekf_node',
            name='ekf_node'
        ),
    ])