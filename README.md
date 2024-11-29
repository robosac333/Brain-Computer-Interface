# Telepathic Navigation
ROS2 Tool that leverages on ML to control a mobile robot using EEG Signals from human brain captured using Muse2 headband and passed the classification filters including ROS2 nodes for being categorized into robot interpretable signals for its locomotions. 

[![Watch the video](https://via.placeholder.com/500)](https://drive.google.com/file/d/11Xi9w9rxSH7TQ3aWLNLaZWAVFkjOsF50/view?usp=sharing)

---

## Overview

The package consists of two main nodes:
- **BCI Node**: Processes brain-computer interface signals and publishes commands
- **Robot Controller Node**: Converts BCI commands into robot movement commands

## Prerequisites

- ROS2 (Humble or newer recommended)
- Python 3.8+
- Ubuntu 22.04 (recommended)

## Package Structure

bci_package/
├── bci_package/
│ ├── init.py
│ ├── bci_node.py # BCI signal processing node
│ └── robot_cont_node.py # Robot controller node
├── package.xml
├── setup.cfg
└── setup.py

## Installation

```

mkdir -p bci_py_ws/src
cd bci_py_ws/src
# clone the repo here

```
## Build the package

```
cd ..
colcon build
source install/setup.bash
```
## Running the package

```
cd ..

# Terminal 1
ros2 run bci_package bci_node

#Terminal
ros2 run bci_package robot_cont_node

```

## Potential ROS2 package Workflow

![Screenshot from 2024-11-27 03-19-23](https://github.com/user-attachments/assets/e3f2f6b0-fd9d-4392-9f0f-13c6598b5c43)

---

## Right hand Sample data
![image](https://github.com/user-attachments/assets/51246e3d-29a7-4829-9b88-26f35c698826)

## Left Hand Sample Data
![image](https://github.com/user-attachments/assets/e3ef8de6-65ec-4b8d-b6c2-66e1d30c2418)
