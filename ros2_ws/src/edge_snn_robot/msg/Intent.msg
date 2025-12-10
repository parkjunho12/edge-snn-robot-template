# Time stamp
std_msgs/Header header

# Classification result
int32 gesture_id          # 0~N
string gesture_name       # "Hand Open", etc.
float32 confidence        # 0.0~1.0

# Performance metrics
float32 latency_ms        # inference time (ms)

# Model metadata
string model_type         # "TCN", "SNN", "Hybrid"
string device             # "CPU", "GPU", "CUDA", etc.
