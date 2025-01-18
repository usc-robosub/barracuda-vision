FROM ros:noetic-ros-base-focal

RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends git vim wget \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/barracuda-camera

# Set working directory
WORKDIR /opt

# Source the workspace on container start
CMD ["/bin/bash", "/opt/barracuda-vision/entrypoint.sh"]