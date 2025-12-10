# Time stamp
std_msgs/Header header

# Joint control
string[] joint_names      # ["wrist_roll", "thumb_1", ...]
float32[] joint_angles    # angles (radians)

# PWM output to hardware (optional)
int32[] pwm_channels      # 0..16
int32[] pwm_values        # 1000..2000 µs

# Gesture metadata
int32 gesture_id
float32 confidence
