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

## 🚀 Quick Start

### Prerequisites

- Ubuntu 22.04
- ROS2 Humble
- Python 3.8+
- FastAPI inference server

### Installation (3 steps)

```bash
# 1. Clone/navigate to workspace
cd ros2_ws

# 2. Run setup script
./setup.sh

# 3. Source workspace
source install/setup.bash
```

### Running the System

```bash
# Terminal 1: Start FastAPI server
cd /path/to/project
./start_server.sh

# Terminal 2: Launch ROS2
ros2 launch edge_snn_robot robot_control.launch.py
```

**That's it!** The system is now running with fake hardware. 🎉

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

**Questions?** Check the Korean guide: `ROS2_GUIDE_KR.md`