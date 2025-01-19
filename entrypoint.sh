#!/usr/bin/bash
source /opt/ros/noetic/setup.bash

cd barracuda-vision/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash

roslaunch barracuda_vision barracuda_vision.launch