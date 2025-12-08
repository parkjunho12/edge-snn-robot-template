#!/usr/bin/env bash
set -e

# ROS 환경 로드
# source /opt/ros/humble/setup.bash

# 워킹 디렉토리로 이동
cd /lab

# uvicorn 실행 (FastAPI 앱 경로 맞추기)
exec uvicorn src.infer_server.app:app --host 0.0.0.0 --port 8000
