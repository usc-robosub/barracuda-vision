FROM ros:noetic-ros-base-focal

RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends git vim wget \
    libboost-all-dev \
    libopencv-dev \
    libx11-dev \
    ros-noetic-cv-bridge \
    ros-noetic-darknet-ros-msgs \
    ros-noetic-image-transport \
    ros-noetic-nodelet \
    ros-noetic-actionlib \
    && rm -rf /var/lib/apt/lists/*

COPY . /opt/barracuda-vision

WORKDIR /opt

# Source the workspace on container start
CMD ["/bin/bash", "/opt/barracuda-vision/entrypoint.sh"]