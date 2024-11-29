#!/usr/bin/env python3

"""
BCI Publisher Node

This node is responsible for publishing BCI (Brain-Computer Interface) data.
It reads brain signals and publishes them to ROS2 topics for other nodes to consume.
"""

# Standard library imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random
import time

class BCIPublisher(Node):
    """
    A ROS2 node that publishes BCI data.
    
    This class handles the publication of brain signal data to other nodes
    in the ROS2 network.
    """

    def __init__(self):
        super().__init__('bci_publisher') # Initialize the BCI publisher node

        # Create publisher with a queue size of 10        
        self.publisher_ = self.create_publisher(String, 'bci_commands', 10)
        
        # Set up timer for periodic publishing
        self.timer = self.create_timer(5.0, self.publish_command)  # Publish every 5 seconds
        
        # Initialize command list
        self.commands = ['move forward', 'move backward', 'move left', 'move right']
        
        # Log node startup
        self.get_logger().info('BCI Publisher Node has started')

    def publish_command(self): # Publish a random BCI command 
        msg = String()
        msg.data = random.choice(self.commands)
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing command: {msg.data}')

def main(args=None): # Main function to initialize and run the BCI publisher node
    rclpy.init(args=args)
    node = BCIPublisher()
    rclpy.spin(node)
    
    # Clean up
    rclpy.shutdown()

if __name__ == '__main__':
    main()