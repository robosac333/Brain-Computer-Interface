#!/usr/bin/env python3

"""
Robot Control Node

This node handles the control of the robot based on received BCI commands.
It subscribes to BCI commands and converts them into robot movement commands
using the ROS2 Twist messages for differential drive robots.
"""

# Standard library imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time

class RobotController(Node):
    """
    A ROS2 node that controls robot movement based on BCI commands.
    
    This class handles the subscription to BCI commands and converts them
    into appropriate robot movement commands using velocity control.
    """

    def __init__(self):
        """Initialize the robot controller node with publishers and subscribers."""
        super().__init__('robot_controller')
        
        # Create subscription to receive BCI commands
        self.subscription = self.create_subscription(
            String,
            'bci_commands',
            self.command_callback,
            10)  # Queue size of 10
        
        # Create publisher for robot velocity commands
        self.velocity_publisher = self.create_publisher(
            Twist,
            'cmd_vel',  # Standard topic for differential drive robots
            10)  # Queue size of 10
        
        # Log node startup
        self.get_logger().info('Robot Controller Node has started')
        
        # Set duration for each movement command
        self.movement_duration = 3.0  # seconds

    def command_callback(self, msg):
        """
        Callback function that processes received BCI commands.
        
        This method converts BCI commands into robot movements by publishing
        appropriate velocity commands for a specified duration.
        
        Args:
            msg: The received message containing the BCI command
        """
                
        command = msg.data
        self.get_logger().info(f'BCI command received: {command}')
        
        # Initialize velocity message
        vel_msg = Twist()
        
        # Configure velocity based on received command
        if command == 'move forward':
            vel_msg.linear.x = 0.2  # Forward velocity in m/s
        elif command == 'move backward':
            vel_msg.linear.x = -0.2  # Backward velocity in m/s
        elif command == 'move left':
            vel_msg.angular.z = 0.5  # Counterclockwise angular velocity in rad/s
        elif command == 'move right':
            vel_msg.angular.z = -0.5  # Clockwise angular velocity in rad/s
        
        # Record start time for movement duration tracking
        start_time = time.time()
        
        # Publish velocity commands for the specified duration
        while time.time() - start_time < self.movement_duration:
            self.velocity_publisher.publish(vel_msg)
            time.sleep(0.1)  # Small delay to control publishing rate
        
        # Stop the robot by publishing zero velocities
        stop_msg = Twist()
        self.velocity_publisher.publish(stop_msg)
        self.get_logger().info('Movement completed')

def main(args=None):
    """
    Main function to initialize and run the robot controller node.
    
    Args:
        args: Command line arguments (default: None)
    """
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()