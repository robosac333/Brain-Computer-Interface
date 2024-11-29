from setuptools import setup
import os
from glob import glob

package_name = 'bci_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Kashif Ansari',
    maintainer_email='kansari@umd.edu',
    description='ROS2 package for BCI control of Nova Carter',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bci_pub_node = bci_package.bci_pub_node:main',
            'robot_cont_node = bci_package.robot_cont_node:main'
        ],
    },
)