FROM ros:noetic-ros-base-focal

RUN sudo apt-get update \
    && sudo apt-get install -y --no-install-recommends git vim wget \
    python3-pip \
    libboost-all-dev \
    libopencv-dev \
    libx11-dev \
    ros-noetic-cv-bridge \
    ros-noetic-darknet-ros-msgs \
    ros-noetic-image-transport \
    ros-noetic-nodelet \
    ros-noetic-image-pipeline \
    ros-noetic-actionlib \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN sudo pip install --no-cache-dir -r requirements.txt

COPY . /opt/barracuda-vision
WORKDIR /opt
# Source the workspace on container start
CMD ["/bin/bash", "/opt/barracuda-vision/entrypoint.sh"]