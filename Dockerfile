FROM ros:noetic-ros-base-focal

RUN apt-get update \
    && apt-get install -y --no-install-recommends git vim wget \
    python3-pip \
    libopencv-dev \
    libx11-dev \
    ros-noetic-cv-bridge \
    ros-noetic-darknet-ros-msgs \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN sudo pip install --no-cache-dir -r requirements.txt

RUN echo "source /opt/barracuda-vision/catkin_ws/devel/setup.bash" >> /root/.bashrc

COPY . /opt/barracuda-vision
WORKDIR /opt
# Source the workspace on container start
CMD ["/bin/bash", "/opt/barracuda-vision/entrypoint.sh"]