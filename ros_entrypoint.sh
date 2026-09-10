#!/bin/bash
set -e

source "/opt/ros/jazzy/setup.bash"
source "/home/ubuntu/ros2_ws/install/setup.bash"

exec "$@"
