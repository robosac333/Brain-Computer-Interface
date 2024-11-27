# Telepathic Navigation
 Software that controls an automotive based on EEG Frequencies in the mind. The live data stream is passed through PCA to get essential information and Decision Tree classifies the required action accordingly

## Potential ROS2 Workspace Structure
```

brain_control_ws/
├── src/
│   ├── brain_control_interfaces/         # Custom message/service definitions if needed
│   │   ├── msg/
│   │   │   └── BrainCommand.msg         # Custom message type for brain signals
│   │   ├── CMakeLists.txt
│   │   └── package.xml
│   │
│   └── brain_control_robot/             # Main C++ package
│       ├── include/
│       │   └── brain_control_robot/
│       │       ├── brain_control_node.hpp
│       │       ├── model/
│       │       │   └── model_interface.hpp   # Interface to ML model
│       │       └── utils/
│       │           └── data_processor.hpp    # Signal processing utilities
│       │
│       ├── src/
│       │   ├── brain_control_node.cpp
│       │   ├── model/
│       │   │   └── model_interface.cpp
│       │   └── utils/
│       │       └── data_processor.cpp
│       │
│       ├── config/
│       │   └── params.yaml               # Configuration parameters
│       │
│       ├── launch/
│       │   └── brain_control.launch.py   # Launch file
│       │
│       ├── model/                        # Directory for ML model files
│       │   └── saved_model/              # Trained model weights/parameters
│       │
│       ├── test/                         # Unit tests
│       │   ├── test_brain_control.cpp
│       │   └── test_main.cpp
│       │
│       ├── CMakeLists.txt
│       ├── package.xml
       └── README.md

```
## Potential ROS2 Workflow

![Screenshot from 2024-11-27 03-19-23](https://github.com/user-attachments/assets/e3f2f6b0-fd9d-4392-9f0f-13c6598b5c43)

---

## Right hand Sample data
![image](https://github.com/user-attachments/assets/51246e3d-29a7-4829-9b88-26f35c698826)

## Left Hand Sample Data
![image](https://github.com/user-attachments/assets/e3ef8de6-65ec-4b8d-b6c2-66e1d30c2418)
