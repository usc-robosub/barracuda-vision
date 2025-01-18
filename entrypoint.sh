#!/usr/bin/bash
source /opt/ros/noetic/setup.bash

# Build catkin_ws
cd barracuda-vision/catkin_ws
catkin_make
source devel/setup.bash

# Start interactive shell session in /opt/barracuda-vision directory
cd ..
exec /bin/bash