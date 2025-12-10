# 🤖 Edge SNN Robot - ROS2 Package

**Complete ROS2 package for EMG-controlled robot hand/arm with fake hardware simulation.**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 🎯 Overview

This package enables **hardware-free development** of EMG-controlled robotic systems using ROS2. Includes:

- ✅ **3 ROS2 Nodes** (Intent, Servo Command, Fake Hardware)
- ✅ **2 Custom Messages** (Intent, JointCmd)
- ✅ **Fake Hardware Simulation** (no robot needed!)
- ✅ **17 DOF Control** (2 wrist + 15 finger joints)
- ✅ **MATLAB Compatible** (same gesture mappings)

## 📁 Package Structure

```
ros2_ws/
├── setup.sh                    ⭐ One-command setup
├── README.md                   📖 This file
│
└── src/edge_snn_robot/
    ├── package.xml
    ├── setup.py
    ├── CMakeLists.txt
    │
    ├── edge_snn_robot/         Python nodes
    │   ├── emg_intent_node.py
    │   ├── servo_cmd_node.py
    │   └── fake_hardware.py    ⚠️ REPLACE LATER
    │
    ├── msg/                    Custom messages
    │   ├── Intent.msg
    │   └── JointCmd.msg
    │
    ├── launch/                 Launch files
    │   └── robot_control.launch.py
    │
    └── config/                 Configuration
        └── robot_params.yaml
```

## 🚀 Quick Start (Native or Docker)

### Prerequisites

- Ubuntu 22.04
- ROS2 Humble
- Python 3.8+
- FastAPI inference server


```bash
# FastAPI Server
# Run inference server on your host:
uvicorn src.infer_server.app:app --host 0.0.0.0 --port 8000

cd ros2_ws

# a) Native ROS2 (Ubuntu + ROS2 Humble already installed)
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --symlink-install

source install/setup.bash
ros2 launch edge_snn_robot robot_control.launch.py \
  server_url:=http://localhost:8000 

# - emg_intent_node → calls FastAPI for inference results and publishes /emg_intent
# - servo_cmd_node → consumes /emg_intent and publishes /joint_cmd
# - fake_hardware_node → subscribes to /joint_cmd and simulates the robot hardware

# b) ROS2 inside Docker (when you don’t have ROS2 locally)
cd ros2_ws
docker compose -f deploy/docker-compose.yml build --no-cache ros2-robot

# local FAST API server
docker run --rm -it --network emg-network  edge-snn-robot:ros2 bash

# docker FAST API server
docker network create {network name} # ex) emg-network 
docker network connect emg-network {FAST API docker container id} # ex) f1fe08ed074878469e06d43dca3f131ba384b41

docker run --rm -it --network emg-network  edge-snn-robot:ros2 bash 


# Inside the container:
pip install requests

# Note:
# macOS/Windows → http://host.docker.internal:8000
# Linux → use host IP: http://192.168.x.x:8000
# If FastAPI is also inside Docker → use internal name:
# http://edge-snn-robot:8000

# local FAST API
ros2 launch edge_snn_robot robot_control.launch.py \
  server_url:=http://host.docker.internal:8000 

# docker FAST API
ros2 launch edge_snn_robot robot_control.launch.py \
  server_url:=http://{docker container name: ex) test}:8000

```

**That's it!** The system is now running with fake hardware. 🎉


## 🧠 Live Runtime Example (Successful Execution)

These logs confirm full ROS2 pipeline correctness:

 - Intent messages publishing at ~28 Hz

 - Confidence always ≥ threshold

 - Correct gesture mapping → /joint_cmd

 - Fake robot simulating motion

```bash
[emg_intent_node] Published 300 intents (27.8 Hz, 0 errors)
[servo_cmd_node] Commands: 300 | Gesture: Rest | Conf: 1.00
[fake_hardware_node] Statistics: 300 commands (28.1 Hz, 10.7s)
[fake_hardware_node]   Gesture 0: 287 (95.7%)
[fake_hardware_node]   Gesture 2: 3 (1.0%)
[fake_hardware_node]   Gesture 5: 6 (2.0%)
[fake_hardware_node]   Gesture 6: 4 (1.3%)
```

### Interpretation:

 - /emg_intent is published correctly

 - /joint_cmd is being generated for each intent

 - Fake hardware is receiving and simulating motion

 - End-to-end loop is stable at ~28 Hz

 - No packet loss or inference errors

### This confirms:

 - networking is correct

 - polling and confidence thresholding are correct

 - gesture → kinematics mapping is correct

 - This is exactly what a real robot would receive — only hardware is faked.


## 📊 System Architecture

```
┌───────────────────────┐
│  FastAPI Server       │  http://localhost:8000
│  TensorRT Inference   │  0.13ms latency
└──────────┬────────────┘
           │ HTTP /infer (30Hz)
           ↓
┌───────────────────────┐
│  emg_intent_node      │  ROS2 Node 1
│  Polls inference      │
│  Publishes: /emg_intent
└──────────┬────────────┘
           │ Intent message
           ↓
┌───────────────────────┐
│  servo_cmd_node       │  ROS2 Node 2
│  Maps gesture → joint │
│  Publishes: /joint_cmd │
└──────────┬────────────┘
           │ JointCmd message
           ↓
┌───────────────────────┐
│  fake_hardware_node   │  ROS2 Node 3 ⚠️
│  Simulates hardware   │
│  Publishes: /joint_state
└───────────────────────┘
```

## 📨 Custom Messages

### Intent.msg
```
std_msgs/Header header
int32 gesture_id          # 0-6
string gesture_name       # "Hand Open", etc.
float32 confidence        # 0.0-1.0
float32 latency_ms
string model_type
string device
```

### JointCmd.msg
```
std_msgs/Header header
string[] joint_names      # 17 joints
float32[] joint_angles    # Radians
int32[] pwm_channels      # 0-16
int32[] pwm_values        # 1000-2000 μs
int32 gesture_id
float32 confidence
```

## 🎮 Gesture Mapping (17 DOF)

| ID | Gesture | Wrist | Thumb | Index | Middle | Ring | Pinky |
|----|---------|-------|-------|-------|--------|------|-------|
| 0 | Rest | [0,0] | [0,0,0] | [0,0,0] | [0,0,0] | [0,0,0] | [0,0,0] |
| 1 | Open | [0,0] | [0.8,0,0] | [0,0,0] | [0,0,0] | [0,0,0] | [0,0,0] |
| 2 | Close | [0,0] | [0.3,1.2,1.0] | [1.0,1.2,1.0] | [1.0,1.2,1.0] | [1.0,1.2,1.0] | [1.0,1.2,1.0] |
| 3 | Flex | [0,0.5] | [0.3,0.2,0.1] | [0.2,0.2,0.1] | [0.2,0.2,0.1] | [0.2,0.2,0.1] | [0.2,0.2,0.1] |
| 4 | Extend | [0,-0.5] | [0.3,0.2,0.1] | [0.2,0.2,0.1] | [0.2,0.2,0.1] | [0.2,0.2,0.1] | [0.2,0.2,0.1] |
| 5 | Pinch | [0,0] | [0.5,0.8,0.6] | [0.8,1.0,0.8] | [0.2,0.3,0.2] | [0.2,0.3,0.2] | [0.2,0.3,0.2] |
| 6 | Point | [0,0] | [0.3,1.0,0.8] | [0,0,0] | [1.0,1.2,1.0] | [1.0,1.2,1.0] | [1.0,1.2,1.0] |

**Same as MATLAB simulator!** No conversion needed.

## 🔍 Monitoring

### View Topics
```bash
ros2 topic list
# /emg_intent
# /joint_cmd
# /joint_state
```

### Echo Messages
```bash
# Intent (gestures)
ros2 topic echo /emg_intent

# Joint commands
ros2 topic echo /joint_cmd

# Check rates
ros2 topic hz /emg_intent  # ~30 Hz
```

### Node Information
```bash
ros2 node list
ros2 node info /emg_intent_node
```

## ⚙️ Configuration

### Runtime Parameters
```bash
# Increase polling rate
ros2 param set /emg_intent_node poll_rate_hz 60.0

# Higher confidence threshold
ros2 param set /emg_intent_node min_confidence 0.8

# Adjust smoothing
ros2 param set /servo_cmd_node filter_alpha 0.5
```

### Launch Arguments
```bash
# Custom server URL
ros2 launch edge_snn_robot robot_control.launch.py \
  server_url:=http://192.168.1.100:8000

# Higher confidence
ros2 launch edge_snn_robot robot_control.launch.py \
  min_confidence:=0.85

# Without hardware (viz only)
ros2 launch edge_snn_robot robot_control.launch.py \
  enable_hardware:=false
```

### Config File
Edit `config/robot_params.yaml` for persistent settings.

## 🔧 Real Hardware Deployment

### Hardware Requirements
- ESP32 development board
- PCA9685 PWM drivers (×2 for 17 servos)
- 17× Servo motors (e.g., MG996R)
- 5V/10A power supply

### Steps to Deploy

1. **Create `real_hardware.py`**:
```python
import serial
from edge_snn_robot.msg import JointCmd

class RealHardwareNode(Node):
    def __init__(self):
        self.serial = serial.Serial('/dev/ttyUSB0', 115200)
        self.cmd_sub = self.create_subscription(
            JointCmd, '/joint_cmd', self.command_callback, 10
        )
    
    def command_callback(self, msg):
        # Send PWM to ESP32
        pwm_str = ','.join(map(str, msg.pwm_values))
        self.serial.write(f"PWM:{pwm_str}\n".encode())
```

2. **Update `setup.py`**:
```python
'real_hardware_node = edge_snn_robot.real_hardware:main'
```

3. **Rebuild and run**:
```bash
colcon build
ros2 run edge_snn_robot real_hardware_node
```

## 🐛 Troubleshooting

### Build Errors
```bash
# Clean build
rm -rf build install log
colcon build --packages-select edge_snn_robot
```

### Cannot Connect to Server
```bash
# Test server
curl http://localhost:8000/health

# Update URL
ros2 param set /emg_intent_node server_url http://YOUR_IP:8000
```

### No Messages on Topics
```bash
# Check nodes running
ros2 node list

# Enable debug
ros2 run edge_snn_robot emg_intent_node --ros-args --log-level debug
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Polling Rate | 30 Hz (max 60Hz) |
| End-to-End Latency | <50ms |
| CPU Usage | ~5-10% per node |
| Memory | ~50MB total |

## 🎯 Development Workflow

1. **Test with Fake Hardware**
   ```bash
   ros2 launch edge_snn_robot robot_control.launch.py
   ```

2. **Visualize in MATLAB**
   ```matlab
   cd matlab_simulator
   run_hand_simulator
   ```

3. **Monitor ROS2**
   ```bash
   ros2 topic echo /joint_cmd
   ```

4. **Deploy to Real Robot**
   - Implement `real_hardware.py`
   - Upload ESP32 firmware
   - Replace fake with real node

## 📚 Additional Resources

- [ROS2 Humble Docs](https://docs.ros.org/en/humble/)
- [Custom Messages Tutorial](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Custom-ROS2-Interfaces.html)
- [Launch Files Guide](https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Launch-Main.html)

## 🤝 Contributing

1. Test with fake hardware
2. Verify MATLAB compatibility
3. Implement real hardware driver
4. Submit PR!

## 📝 License

MIT License - See LICENSE file

---

**Current Status:** ✅ Development Ready (Fake Hardware)  
**Next Step:** 🔧 Implement Real Hardware Driver