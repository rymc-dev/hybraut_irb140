# LEGO brick detection

Real-time detection of LEGO bricks for the IRB140 sorting task, using a
**local Ultralytics YOLO** model. `notebooks/trainin.ipynb` is the training
scratch area; the runtime component is the ROS 2 node
`hybraut_irb140/lego_detector.py`.

## Node: `hybraut_irb140_lego_detector`

Mirrors `line_detector.py`: subscribes to the end-effector RGB + depth
camera, runs YOLO per frame, and republishes plain topics for the sorting
automaton.

### Prerequisites

```bash
pip install ultralytics          # pulls torch; CUDA build optional (see device param)
```

`ultralytics` / `torch` are **pip** deps, not rosdep-managed.

### Weights

Drop a trained model at `lego_detection/weights/lego_yolo.pt` **before**
`colcon build` (it is installed to
`share/hybraut_irb140/weights/lego_yolo.pt` and picked up automatically), or
pass an absolute path at launch:

```bash
ros2 run hybraut_irb140 hybraut_irb140_lego_detector \
  --ros-args -p model_path:=/abs/path/to/lego_yolo.pt
```

`*.pt` files are git-ignored. Training source referenced in the
dissertation: the Kaggle `b100-lego-detection-dataset` (YOLO format). Class
names are read from the `.pt` itself — nothing is hard-coded.

### Run (against the sim camera)

```bash
ros2 run hybraut_irb140 hybraut_irb140_lego_detector --ros-args \
  -p rgb_topic:=/rgbd_camera/image \
  -p depth_topic:=/rgbd_camera/depth_image \
  -p camera_info_topic:=/rgbd_camera/camera_info
```

### Topics

| Topic | Type | Notes |
|---|---|---|
| `/hybraut/hybraut_irb140/lego_detections` | `vision_msgs/Detection2DArray` | pixel boxes + class + score, header in the camera frame |
| `/hybraut/hybraut_irb140/lego_detections_3d` | `vision_msgs/Detection3DArray` | brick centre poses in `base_frame`; box size is a rough depth/focal estimate |
| `/hybraut/hybraut_irb140/lego_detections/image` | `sensor_msgs/Image` | annotated debug image (`publish_annotated`) |

Both detection arrays are published every processed frame (empty when
nothing is seen), so a consumer can tell "no bricks" from "node dead".

### Parameters

| Param | Default | Meaning |
|---|---|---|
| `rgb_topic` | `/camera/color/image_raw` | RGB input |
| `depth_topic` | `/camera/depth/image_rect_raw` | depth input, assumed pixel-aligned to RGB |
| `camera_info_topic` | `/camera/color/camera_info` | intrinsics (cached) |
| `base_frame` | `base_link` | TF target for 3D poses |
| `model_path` | `""` | empty → bundled `weights/lego_yolo.pt`; else an absolute `.pt` path |
| `device` | `cpu` | `cpu`, or `0` / `cuda:0` with a CUDA torch build |
| `confidence_threshold` | `0.25` | YOLO `conf` |
| `iou_threshold` | `0.45` | YOLO NMS `iou` |
| `imgsz` | `640` | inference size |
| `max_detections` | `20` | YOLO `max_det` |
| `class_whitelist` | `[]` | if non-empty, keep only these class names |
| `depth_window_px` | `1` | radius of the median depth window (`1` → 3×3) |
| `publish_annotated` | `true` | publish the drawn-boxes image |
| `min_score_for_3d` | `0.0` | low-score dets still appear in 2D; gate 3D back-projection |
| `tf_timeout` | `0.1` | seconds to wait for the camera→`base_frame` TF at the image stamp |

### Notes / limitations

- CPU inference at 640 px is a few hundred ms/frame. A drop-frame guard
  means the effective rate is inference-bound and frames are dropped, not
  buffered — use a CUDA torch build + `device:=0` for real-time.
- If the depth image is not aligned/sized to the RGB image, affected
  detections simply get no 3D pose (2D output is unaffected).
- Run with `-p use_sim_time:=true` against the sim so the image stamps and
  the TF tree share Gazebo's clock. If the stamped camera→`base_frame`
  lookup still fails with "extrapolation into the future" (the image stamp
  landing a few ms ahead of the last TF broadcast), the node retries once
  against the latest available transform - fine while the arm is roughly
  stationary over a pick, slightly stale if it is moving fast.
- 3D box `size` assumes a roughly fronto-parallel brick face; treat it as
  indicative, not metric-accurate.
