#!/bin/bash

################################################################################
# Jetson Setup Script for EMG Inference System
# Prepares NVIDIA Jetson device for edge deployment
# Supports: Jetson Nano, TX2, Xavier NX, AGX Xavier, Orin
################################################################################

set -e  # Exit on error

echo "=========================================="
echo "Jetson Edge Device Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${RED}Error: This script must be run on an NVIDIA Jetson device${NC}"
    exit 1
fi

# Get Jetson info
JETSON_MODEL=$(cat /etc/nv_tegra_release | grep -oP 'BOARD: \K[^,]+' || echo "Unknown")
JETPACK_VERSION=$(cat /etc/nv_tegra_release | grep -oP 'R\d+' || echo "Unknown")

echo "Detected Jetson: $JETSON_MODEL"
echo "JetPack Version: $JETPACK_VERSION"
echo ""

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}Warning: Some operations require sudo privileges${NC}"
    echo "You may be prompted for your password"
    echo ""
fi

# Update system
echo "Step 1/8: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker if not present
echo ""
echo "Step 2/8: Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo -e "${GREEN}Docker installed successfully${NC}"
else
    echo -e "${GREEN}Docker already installed${NC}"
fi

# Install NVIDIA Container Runtime
echo ""
echo "Step 3/8: Installing NVIDIA Container Runtime..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify NVIDIA runtime
echo ""
echo "Step 4/8: Verifying NVIDIA Docker runtime..."
if sudo docker run --rm --runtime nvidia nvcr.io/nvidia/l4t-base:r32.7.1 nvidia-smi; then
    echo -e "${GREEN}NVIDIA Docker runtime working${NC}"
else
    echo -e "${YELLOW}Warning: NVIDIA Docker runtime test failed${NC}"
fi

# Install Python dependencies
echo ""
echo "Step 5/8: Installing Python dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-opencv

# Install PyTorch for Jetson
echo ""
echo "Step 6/8: Installing PyTorch for Jetson..."
if ! python3 -c "import torch" &> /dev/null; then
    echo "Downloading PyTorch wheel for Jetson..."
    # Note: Use appropriate PyTorch version for your JetPack
    # For JetPack 5.x (Orin):
    # wget https://nvidia.box.com/shared/static/...torch-2.0.0-cp38-linux_aarch64.whl
    # pip3 install torch-2.0.0-cp38-linux_aarch64.whl
    
    echo -e "${YELLOW}Please install PyTorch manually using NVIDIA's provided wheels${NC}"
    echo "Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
else
    echo -e "${GREEN}PyTorch already installed${NC}"
fi

# Set power mode to maximum
echo ""
echo "Step 7/8: Setting power mode..."
sudo nvpmodel -m 0  # Maximum performance
sudo jetson_clocks  # Enable maximum clocks
echo -e "${GREEN}Power mode set to maximum performance${NC}"

# Create necessary directories
echo ""
echo "Step 8/8: Creating project directories..."
mkdir -p ~/emg-inference/{output/rate,logs,data,config}
echo -e "${GREEN}Directories created${NC}"

# Print system info
echo ""
echo "=========================================="
echo "System Information:"
echo "=========================================="
echo "Jetson Model: $JETSON_MODEL"
echo "JetPack: $JETPACK_VERSION"
echo "Docker: $(docker --version)"
echo "Python: $(python3 --version)"
echo "CUDA: $(nvcc --version | grep release | awk '{print $5}' | sed 's/,//')"
echo ""

# Check available resources
echo "Available Resources:"
echo "- CPU Cores: $(nproc)"
echo "- RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'Unable to detect')"
echo ""

# Print next steps
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Clone your project repository"
echo "2. Copy TensorRT engine to output/rate/"
echo "3. Build Docker image: docker-compose build"
echo "4. Start container: docker-compose up -d"
echo "5. Check logs: docker-compose logs -f"
echo ""
echo -e "${GREEN}Jetson is ready for edge deployment!${NC}"
echo ""
echo "Note: You may need to log out and back in for Docker group changes to take effect"