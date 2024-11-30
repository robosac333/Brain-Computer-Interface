import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import subprocess
import sys

class BCIPublisher(Node):
    def __init__(self):
        super().__init__('bci_publisher')
        self.publisher = self.create_publisher(
            String,
            'bci_predictions',
            10
        )
        self.get_logger().info('BCI Publisher Node started')
        self.publish_predictions()

    def publish_predictions(self):
        try:
            # Run detection.py and capture its output
            script_path = '/home/abhishek/Brain-Computer-Interface/bci_py_ws/src/bci_package/bci_package/detection.py'
            result = subprocess.run(['python3', script_path], 
                                 capture_output=True, 
                                 text=True)
            
            # Split the output into lines
            predictions = result.stdout.strip().split('\n')
            
            # Publish each prediction
            for pred in predictions:
                if "Hand" in pred:  # Only publish lines containing predictions
                    msg = String()
                    msg.data = pred.strip()
                    self.publisher.publish(msg)
                    self.get_logger().info(f'Published: {msg.data}')
                    time.sleep(1)  # Add delay between publications

        except Exception as e:
            self.get_logger().error(f'Error publishing predictions: {str(e)}')
            self.get_logger().error(f'Script output: {result.stderr}')  # Print error output if any
        finally:
            self.get_logger().info('Finished publishing predictions')
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    bci_publisher = BCIPublisher()
    
    try:
        rclpy.spin(bci_publisher)
    except KeyboardInterrupt:
        print('Stopped by keyboard interrupt')
    finally:
        bci_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()