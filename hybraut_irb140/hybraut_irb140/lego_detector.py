#!/usr/bin/env python3
"""
lego_detector.py

Perception node: runs a local Ultralytics YOLO model on the
end-effector-mounted RGB camera to detect LEGO bricks, back-projects each
box centre through the aligned depth image, and republishes the results as
plain topics for the sorting automaton to consume - mirrors the
`line_detector.py` split (a standalone perception node publishing plain
topics that the automaton node subscribes into its auxiliary state).

Outputs:
  * `vision_msgs/Detection2DArray` - pixel boxes + class + score
  * `vision_msgs/Detection3DArray` - brick centre poses in `base_frame`
  * `sensor_msgs/Image`            - annotated debug image (optional)

Camera topic names default to a RealSense-D405-style layout but are all ROS
parameters (the sim `rgbd_camera` publishes `/rgbd_camera/{image,depth_image,
camera_info}`).

Requires `ultralytics` (and its `torch`) on the Python path - install with
`pip install ultralytics`; not a rosdep.
"""

import os
from typing import Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from message_filters import ApproximateTimeSynchronizer, Subscriber

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401 - registers PointStamped transform support

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)

from hybraut_irb140.perception_geometry import backproject_pixel, depth_to_meters


class LegoDetector(Node):

    def __init__(self) -> None:
        super().__init__("lego_detector", namespace="hybraut")

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("model_path", "")
        # dynamic typing: accept both `device:=cpu` (string) and the bare
        # `device:=0` GPU index (parsed as an int on the CLI) - coerced to
        # str below since Ultralytics takes "0" / "cuda:0" / "cpu" all the same.
        self.declare_parameter(
            "device", "cpu", ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("max_detections", 20)
        self.declare_parameter("class_whitelist", [""])
        self.declare_parameter("depth_window_px", 1)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("min_score_for_3d", 0.0)
        self.declare_parameter("tf_timeout", 0.1)

        self._base_frame: str = self.get_parameter("base_frame").value
        self._device: str = str(self.get_parameter("device").value)
        self._conf: float = self.get_parameter("confidence_threshold").value
        self._iou: float = self.get_parameter("iou_threshold").value
        self._imgsz: int = self.get_parameter("imgsz").value
        self._max_det: int = self.get_parameter("max_detections").value
        self._depth_window: int = self.get_parameter("depth_window_px").value
        self._publish_annotated: bool = self.get_parameter("publish_annotated").value
        self._min_score_for_3d: float = self.get_parameter("min_score_for_3d").value
        self._tf_timeout: float = self.get_parameter("tf_timeout").value

        whitelist = self.get_parameter("class_whitelist").value or []
        self._class_whitelist = {name for name in whitelist if name}

        self._model = self._load_model()
        self._class_names = self._model.names  # dict: id -> name

        self._bridge = CvBridge()
        self._camera_model = PinholeCameraModel()
        self._have_camera_info = False
        # drop-frame guard: CPU inference latency must not queue up frames.
        self._busy = False

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._det2d_pub = self.create_publisher(
            Detection2DArray, "/hybraut/hybraut_irb140/lego_detections",
            qos_profile_system_default,
        )
        self._det3d_pub = self.create_publisher(
            Detection3DArray, "/hybraut/hybraut_irb140/lego_detections_3d",
            qos_profile_system_default,
        )
        self._annotated_pub = None
        if self._publish_annotated:
            self._annotated_pub = self.create_publisher(
                Image, "/hybraut/hybraut_irb140/lego_detections/image",
                qos_profile_system_default,
            )

        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._camera_info_cb, qos_profile_system_default,
        )

        # rgb/depth need to be from the same instant - camera_info is
        # effectively static (just intrinsics), so it's cached separately
        # rather than folded into the synchronizer.
        self._rgb_sub = Subscriber(self, Image, self.get_parameter("rgb_topic").value)
        self._depth_sub = Subscriber(self, Image, self.get_parameter("depth_topic").value)
        self._sync = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._image_cb)

        self.get_logger().info(
            f"lego_detector ready - model classes: {list(self._class_names.values())}"
        )

    def _load_model(self):
        """Resolve the weights path and load the Ultralytics model, or exit."""
        from ultralytics import YOLO

        model_path: str = self.get_parameter("model_path").value
        if not model_path:
            model_path = os.path.join(
                get_package_share_directory("hybraut_irb140"), "weights", "lego_yolo.pt"
            )
        if not os.path.isfile(model_path):
            self.get_logger().fatal(
                f"YOLO weights not found at '{model_path}' - set the 'model_path' "
                f"parameter to a trained .pt, or drop 'lego_yolo.pt' into the "
                f"package's weights/ dir."
            )
            raise SystemExit(1)

        self.get_logger().info(f"loading YOLO weights: {model_path} (device={self._device})")
        model = YOLO(model_path)
        try:
            model.fuse()
        except Exception:  # noqa: BLE001 - fuse is a best-effort speed-up only
            pass
        return model

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._camera_model.fromCameraInfo(msg)
        self._have_camera_info = True

    def _image_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        if not self._have_camera_info or self._busy:
            return
        self._busy = True
        try:
            self._process(rgb_msg, depth_msg)
        finally:
            self._busy = False

    def _process(self, rgb_msg: Image, depth_msg: Image) -> None:
        bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = depth_to_meters(
            self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        )

        result = self._model.predict(
            bgr, conf=self._conf, iou=self._iou, imgsz=self._imgsz,
            max_det=self._max_det, device=self._device, verbose=False,
        )[0]

        det2d_array = Detection2DArray()
        det2d_array.header = rgb_msg.header
        det3d_array = Detection3DArray()
        det3d_array.header.frame_id = self._base_frame
        det3d_array.header.stamp = rgb_msg.header.stamp

        fx, fy = self._camera_model.fx(), self._camera_model.fy()

        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            name = self._class_names.get(class_id, str(class_id))
            if self._class_whitelist and name not in self._class_whitelist:
                continue

            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)
            w_px = x2 - x1
            h_px = y2 - y1

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = name
            hypothesis.hypothesis.score = score

            det2d = Detection2D()
            det2d.header = rgb_msg.header
            det2d.bbox = BoundingBox2D()
            det2d.bbox.center.position.x = u
            det2d.bbox.center.position.y = v
            det2d.bbox.size_x = w_px
            det2d.bbox.size_y = h_px
            det2d.results.append(hypothesis)
            det2d_array.detections.append(det2d)

            if score < self._min_score_for_3d:
                continue

            pt_cam = backproject_pixel(
                u, v, depth, self._camera_model, window=self._depth_window
            )
            if pt_cam is None:
                self.get_logger().debug(f"no valid depth for '{name}' box at ({u:.0f}, {v:.0f})")
                continue
            z = pt_cam.point.z  # camera-frame range, before the frame change
            pt_base = self._transform_point(pt_cam, rgb_msg.header)
            if pt_base is None:
                continue

            det3d = Detection3D()
            det3d.header = det3d_array.header
            det3d.bbox.center.position.x = pt_base.point.x
            det3d.bbox.center.position.y = pt_base.point.y
            det3d.bbox.center.position.z = pt_base.point.z
            det3d.bbox.center.orientation.w = 1.0
            # metric box size is a rough estimate: pixel extent scaled by
            # depth / focal length, assuming the brick face is ~fronto-parallel.
            det3d.bbox.size.x = w_px * z / fx
            det3d.bbox.size.y = h_px * z / fy
            det3d.bbox.size.z = min(det3d.bbox.size.x, det3d.bbox.size.y)

            hyp3d = ObjectHypothesisWithPose()
            hyp3d.hypothesis.class_id = name
            hyp3d.hypothesis.score = score
            hyp3d.pose.pose.position.x = pt_base.point.x
            hyp3d.pose.pose.position.y = pt_base.point.y
            hyp3d.pose.pose.position.z = pt_base.point.z
            hyp3d.pose.pose.orientation.w = 1.0
            det3d.results.append(hyp3d)
            det3d_array.detections.append(det3d)

        self._det2d_pub.publish(det2d_array)
        self._det3d_pub.publish(det3d_array)

        if self._annotated_pub is not None:
            annotated = result.plot()  # BGR ndarray with boxes + labels drawn
            annotated_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_msg.header = rgb_msg.header
            self._annotated_pub.publish(annotated_msg)

    def _transform_point(self, point: PointStamped, header) -> Optional[PointStamped]:
        # The eye-in-hand camera moves with the arm, so the lookup at the
        # exact capture time is the correct one. In sim the image stamp often
        # lands a few ms ahead of the last TF broadcast ("extrapolation into
        # the future") - fall back to the latest available transform, which
        # is close enough while the arm is near-stationary over a pick.
        for stamp in (header.stamp, Time().to_msg()):
            point.header.stamp = stamp
            try:
                return self._tf_buffer.transform(
                    point, self._base_frame,
                    timeout=Duration(seconds=self._tf_timeout),
                )
            except tf2_ros.ExtrapolationException:
                continue  # retry against the latest available transform
            except tf2_ros.TransformException as e:
                self.get_logger().warning(
                    f"could not transform brick point into '{self._base_frame}': {e}"
                )
                return None
        self.get_logger().warning(
            f"could not transform brick point into '{self._base_frame}': "
            f"no usable TF at the image stamp or latest"
        )
        return None


def main() -> None:
    rclpy.init()
    node = LegoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
