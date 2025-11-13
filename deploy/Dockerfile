# 베이스: ROS 2 Humble (Ubuntu 22.04)
FROM osrf/ros:humble-desktop

# 시스템 패키지 & rclpy 설치 (apt)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip python3-venv python3-dev \
      build-essential \
      ros-humble-rclpy \
    && rm -rf /var/lib/apt/lists/*

# 가상환경 생성 및 PATH 설정
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 2. python3 명령을 python으로 연결 (GitHub Actions용 편의)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# 3. CI용 툴 포함 (테스트 도구)
RUN pip3 install -U pip flake8 pytest

# 파이썬 업그레이드 & 워킹 디렉토리
ENV PYTHONUNBUFFERED=1
WORKDIR /lab

# (컨텍스트를 리포 루트로 빌드한다고 가정) 의존성 설치
COPY requirements.txt /lab/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /lab/requirements.txt

# 앱 소스 복사
COPY src /lab/src

# 기본 포트
EXPOSE 8000

# 엔트리포인트 복사 및 권한 부여
COPY deploy/entrypoint.sh /lab/deploy/entrypoint.sh
RUN chmod +x /lab/deploy/entrypoint.sh

# 비루트 유저 생성 및 폴더 소유권 변경
RUN useradd -ms /bin/bash appuser && chown -R appuser:appuser /lab
USER appuser

ENTRYPOINT ["/bin/bash", "/lab/deploy/entrypoint.sh"]
