# syntax=docker/dockerfile:1
FROM ros:jazzy-ros-base-noble

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# --- OS tooling ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      vim \
      git \
      python3-pip \
    && rm -rf /var/lib/apt/lists/*

ENV WS=/home/ubuntu/ros2_ws
WORKDIR $WS

# --- ROS dependencies (resolved from package.xml via rosdep) ---
# Copy only the manifest first so this layer is cached until the deps change.
COPY hybraut_irb140/package.xml src/hybraut_irb140/package.xml
RUN apt-get update \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src -y \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (packages with no rosdep key) ---
# Do NOT `apt-get remove python3-numpy`: it is a dependency of ros-jazzy-ros2cli
# and ~40 other ROS packages, so removing it uninstalls the `ros2` CLI.
# constraints.txt pins numpy < 2 so pip keeps the apt-managed numpy 1.26.4 (which
# ROS's C extensions are built against) instead of trying to upgrade it - an
# upgrade fails at build time ("Cannot uninstall numpy 1.26.4, RECORD file not
# found") and breaks cv_bridge at runtime with a numpy ABI error.
# CPU-only torch/torchvision: the base image has no CUDA runtime, so the default
# CUDA build would add several GB for nothing. Switch the base image and drop the
# --index-url override if GPU inference is ever needed.
COPY hybraut_irb140/requirements.txt src/hybraut_irb140/requirements.txt
COPY hybraut_irb140/constraints.txt src/hybraut_irb140/constraints.txt
RUN pip install --no-cache-dir --break-system-packages \
      --constraint src/hybraut_irb140/constraints.txt \
      --index-url https://download.pytorch.org/whl/cpu \
      torch torchvision
RUN pip install --no-cache-dir --break-system-packages \
      --constraint src/hybraut_irb140/constraints.txt \
      -r src/hybraut_irb140/requirements.txt

# --- Build the workspace ---
COPY hybraut_irb140/ src/hybraut_irb140/
RUN source /opt/ros/jazzy/setup.bash && colcon build --symlink-install

# --- Entrypoint ---
COPY ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["ros2", "run", "hybraut_irb140", "hybraut_irb140_ball_detector"]
