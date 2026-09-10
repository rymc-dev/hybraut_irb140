# hybraut_irb140
Hybrid automaton designed for irb140 move it control

## Running in Docker

The package ships a `Dockerfile` (ROS 2 Jazzy) and a helper script,
`docker_build_run.sh`, that builds the image and runs it on the **host's ROS 2
network** so nodes in the container discover nodes running on the host (and vice
versa).

### Quick start

```bash
# Build the image, then run the default node (ball detector)
./docker_build_run.sh

# Build, then drop into an interactive shell in the container
./docker_build_run.sh bash

# Run an arbitrary command (image is rebuilt unless SKIP_BUILD=1)
./docker_build_run.sh ros2 topic list
SKIP_BUILD=1 ./docker_build_run.sh ros2 run hybraut_irb140 hybraut_irb140_line_detector
```

Manual equivalent:

```bash
docker build -t hybraut_irb140:latest .

docker run --rm -it \
  --network host \
  --ipc host \
  --pid host \
  -e ROS_DOMAIN_ID=0 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e ROS_LOCALHOST_ONLY=0 \
  hybraut_irb140:latest
```

### Why these flags

| Flag | Purpose |
| --- | --- |
| `--network host` | Container shares the host's network interfaces, so DDS multicast discovery reaches host nodes. |
| `--ipc host` | Shares the memory namespace so the DDS shared-memory transport works across the host/container boundary. |
| `--pid host` | Lets `ros2 node list` / `ros2 doctor` see host processes (optional). |
| `-e ROS_DOMAIN_ID` | Must match the host (default `0`). |
| `-e RMW_IMPLEMENTATION` | Must match the host's middleware (default `rmw_cyclonedds_cpp`). |
| `-e ROS_LOCALHOST_ONLY=0` | Allow discovery beyond localhost. |

### Configuration

The script reads these environment variables (all optional):

| Variable | Default | Meaning |
| --- | --- | --- |
| `IMAGE` | `hybraut_irb140:latest` | Image tag to build/run. |
| `CONTAINER` | `hybraut_irb140` | Container name. |
| `ROS_DOMAIN_ID` | `0` | ROS 2 domain ID (match the host). |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | DDS vendor (match the host). |
| `SKIP_BUILD` | `0` | Set to `1` to skip `docker build` and reuse the existing image. |

Example:

```bash
ROS_DOMAIN_ID=7 RMW_IMPLEMENTATION=rmw_fastrtps_cpp ./docker_build_run.sh
```

### Middleware must match the host

The `ros:jazzy-ros-base-noble` base image only includes **Fast DDS**
(`rmw_fastrtps_cpp`). If the host runs Cyclone DDS (`rmw_cyclonedds_cpp`), pick
one of:

- **Add Cyclone to the image** — add `ros-jazzy-rmw-cyclonedds-cpp` to the
  `apt-get install` list in the `Dockerfile`, then rebuild. The script already
  defaults `RMW_IMPLEMENTATION` to `rmw_cyclonedds_cpp`.
- **Use Fast DDS on both sides** — run with
  `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` and export the same on the host.

Host and container will not discover each other if their `RMW_IMPLEMENTATION`
(or `ROS_DOMAIN_ID`) differ.

### Available nodes

```
ros2 run hybraut_irb140 hybraut_irb140                    # hybrid automaton
ros2 run hybraut_irb140 hybraut_irb140_line_detector
ros2 run hybraut_irb140 hybraut_irb140_line_follower
ros2 run hybraut_irb140 hybraut_irb140_lego_detector
ros2 run hybraut_irb140 hybraut_irb140_ball_detector      # default CMD
```
