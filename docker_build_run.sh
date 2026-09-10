#!/usr/bin/env bash
# Build the hybraut_irb140 image and run it sharing the host's ROS 2 network.
#
# Usage:
#   ./docker_build_run.sh              # build, then run the default CMD (ball detector)
#   ./docker_build_run.sh bash         # build, then drop into an interactive shell
#   ./docker_build_run.sh ros2 topic list
#   SKIP_BUILD=1 ./docker_build_run.sh # run only, reuse the existing image
set -euo pipefail

IMAGE="${IMAGE:-hybraut_irb140:latest}"
CONTAINER="${CONTAINER:-hybraut_irb140}"

# Match the host's ROS 2 middleware config so DDS discovery works across the
# host<->container boundary. Override by exporting these before running.
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

cd "$(dirname "$0")"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo ">> Building ${IMAGE}"
  docker build -t "${IMAGE}" .
fi

# --network host  : share the host network namespace (same interfaces/ports)
# --ipc host       : share memory namespace so DDS shared-memory transport works
# --pid host       : optional, lets `ros2 node`/`ros2 doctor` see host PIDs
echo ">> Running ${IMAGE} (ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, RMW=${RMW_IMPLEMENTATION})"
exec docker run --rm -it \
  --name "${CONTAINER}" \
  --network host \
  --ipc host \
  --pid host \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID}" \
  -e RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
  -e ROS_LOCALHOST_ONLY=0 \
  "${IMAGE}" "$@"
