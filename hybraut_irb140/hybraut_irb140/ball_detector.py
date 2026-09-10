#!/usr/bin/env python3
"""
ball_detector.py

Perception node: runs a local Ultralytics YOLO model on the
end-effector-mounted RGB camera to detect balls, back-projects each box
centre through the aligned depth image to a 3D point, and publishes it for
visualisation and downstream use - mirrors `lego_detector.py` (same RGB +
depth sync, `PinholeCameraModel` back-projection and tf2 handling) and adds
a TF broadcast plus RViz spheres.

Outputs:
  * `sensor_msgs/Image`              - the RGB frame with the boxes drawn
  * `visualization_msgs/MarkerArray` - a translucent SPHERE per ball, in
                                       `base_frame`, for RViz
  * `vision_msgs/Detection3DArray`   - ball centre poses in `base_frame`
  * tf                               - a `base_frame` -> `ball` transform
                                       per detected ball

Camera topic names default to the sim's RealSense-style bridge layout
(`/camera/{color/image_raw,depth/image_rect_raw,color/camera_info}`) but are
all ROS parameters.

COCO-pretrained weights already detect class `sports ball`, so `model_path`
may be left empty: the node then loads a packaged `ball_yolo.pt` if present,
else falls back to `yolov8n.pt` (auto-downloaded by Ultralytics). Point
`model_path` at a fine-tuned single-class model for better range/robustness.

Requires `ultralytics` (and its `torch`) on the Python path - install with
`pip install ultralytics`; not a rosdep.
"""

import os
from typing import List, Optional

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from message_filters import ApproximateTimeSynchronizer, Subscriber

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
import tf2_geometry_msgs  # noqa: F401 - registers PointStamped transform support

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, TransformStamped
from vision_msgs.msg import (
    Detection3D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from hybraut_irb140.perception_geometry import backproject_pixel, depth_to_meters


class BallDetector(Node):

    def __init__(self) -> None:
        super().__init__("ball_detector", namespace="hybraut")

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
        self.declare_parameter("confidence_threshold", 0.15)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("max_detections", 10)
        self.declare_parameter("target_classes", ["sports ball", "ball"])
        self.declare_parameter("depth_window_px", 2)
        self.declare_parameter("min_score_for_3d", 0.0)
        self.declare_parameter("tf_frame", "ball")
        self.declare_parameter("marker_ns", "ball")
        self.declare_parameter("marker_lifetime", 0.5)
        self.declare_parameter("marker_alpha", 0.5)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("publish_detections_3d", True)
        self.declare_parameter("tf_timeout", 0.1)

        self._base_frame: str = self.get_parameter("base_frame").value
        self._device: str = str(self.get_parameter("device").value)
        self._conf: float = self.get_parameter("confidence_threshold").value
        self._iou: float = self.get_parameter("iou_threshold").value
        self._imgsz: int = self.get_parameter("imgsz").value
        self._max_det: int = self.get_parameter("max_detections").value
        self._depth_window: int = self.get_parameter("depth_window_px").value
        self._min_score_for_3d: float = self.get_parameter("min_score_for_3d").value
        self._tf_frame: str = self.get_parameter("tf_frame").value
        self._marker_ns: str = self.get_parameter("marker_ns").value
        self._marker_lifetime: float = self.get_parameter("marker_lifetime").value
        self._marker_alpha: float = self.get_parameter("marker_alpha").value
        self._publish_annotated: bool = self.get_parameter("publish_annotated").value
        self._publish_markers: bool = self.get_parameter("publish_markers").value
        self._publish_det3d: bool = self.get_parameter("publish_detections_3d").value
        self._tf_timeout: float = self.get_parameter("tf_timeout").value

        targets = self.get_parameter("target_classes").value or []
        self._target_classes = {name for name in targets if name}

        self._model = self._load_model()
        self._class_names = self._model.names  # dict: id -> name

        self._bridge = CvBridge()
        self._camera_model = PinholeCameraModel()
        self._have_camera_info = False
        # drop-frame guard: inference latency must not queue up frames.
        self._busy = False
        # how many SPHERE markers the previous frame published, so a quieter
        # frame can DELETE the ones left dangling in RViz.
        self._prev_marker_count = 0

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._det3d_pub = None
        if self._publish_det3d:
            self._det3d_pub = self.create_publisher(
                Detection3DArray, "/hybraut/hybraut_irb140/ball_detections_3d",
                qos_profile_system_default,
            )
        self._marker_pub = None
        if self._publish_markers:
            self._marker_pub = self.create_publisher(
                MarkerArray, "/hybraut/hybraut_irb140/ball_markers",
                qos_profile_system_default,
            )
        self._annotated_pub = None
        if self._publish_annotated:
            self._annotated_pub = self.create_publisher(
                Image, "/hybraut/hybraut_irb140/ball_detections/image",
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
            f"ball_detector ready - model classes: {list(self._class_names.values())}; "
            f"keeping: {sorted(self._target_classes) or 'all'}"
        )

    def _load_model(self):
        """Resolve the weights path and load the Ultralytics model.

        Unlike `lego_detector.py`, an empty `model_path` is not fatal: balls
        are a COCO class, so we fall back to the pretrained `yolov8n.pt`
        (Ultralytics downloads it on first use).
        """
        from ultralytics import YOLO

        model_path: str = self.get_parameter("model_path").value
        if not model_path:
            packaged = os.path.join(
                get_package_share_directory("hybraut_irb140"), "weights", "ball_yolo.pt"
            )
            model_path = packaged if os.path.isfile(packaged) else "yolov8n.pt"

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

        # gather the boxes we care about first, so we know how to name the
        # TF frames (bare `ball` when there is only one).
        balls = []  # (x1, y1, x2, y2, score, name)
        for box in result.boxes:
            name = self._class_names.get(int(box.cls[0]), str(int(box.cls[0])))
            if self._target_classes and name not in self._target_classes:
                continue
            score = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            balls.append((x1, y1, x2, y2, score, name))

        overlay = bgr.copy()
        det3d_array = Detection3DArray()
        det3d_array.header.frame_id = self._base_frame
        det3d_array.header.stamp = rgb_msg.header.stamp
        markers = MarkerArray()
        transforms: List[TransformStamped] = []

        fx = self._camera_model.fx()
        single = len(balls) == 1

        published = 0
        for x1, y1, x2, y2, score, name in balls:
            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)
            r_px = 0.25 * ((x2 - x1) + (y2 - y1))  # mean of the two half-extents

            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(overlay, (int(u), int(v)), 3, (0, 255, 0), -1)
            label = f"{name} {score:.2f}"

            pt_cam = backproject_pixel(
                u, v, depth, self._camera_model, window=self._depth_window
            )
            if pt_cam is None or score < self._min_score_for_3d:
                if pt_cam is None:
                    self.get_logger().debug(
                        f"no valid depth for '{name}' box at ({u:.0f}, {v:.0f})"
                    )
                self._draw_label(overlay, label, x1, y1)
                continue

            z = pt_cam.point.z  # range to the near surface, before the frame change
            r_m = r_px * z / fx
            # the box-centre ray strikes the front of the sphere; nudge the
            # point back one radius (ray ~ optical axis for a centred ball).
            pt_cam.point.z += r_m

            # `backproject_pixel` returns REP-103 optical coords (x right, y
            # down, z forward), but this robot's URDF leaves
            # `depth_camera_optical` un-rotated from the camera body link, so
            # tf2 treats its axes as body-style (x forward, y left, z up) -
            # matching the gz point cloud. Re-express the point in those axes
            # here, otherwise the forward range lands on base +Z instead of
            # base +X.
            ox, oy, oz = pt_cam.point.x, pt_cam.point.y, pt_cam.point.z
            pt_cam.point.x = oz
            pt_cam.point.y = -ox
            pt_cam.point.z = -oy

            pt_base = self._transform_point(pt_cam, rgb_msg.header)
            if pt_base is None:
                self._draw_label(overlay, label, x1, y1)
                continue

            self._draw_label(overlay, f"{label}  {z:.2f}m", x1, y1)

            diameter = max(2.0 * r_m, 1e-3)

            if self._det3d_pub is not None:
                det3d = Detection3D()
                det3d.header = det3d_array.header
                det3d.bbox.center.position.x = pt_base.point.x
                det3d.bbox.center.position.y = pt_base.point.y
                det3d.bbox.center.position.z = pt_base.point.z
                det3d.bbox.center.orientation.w = 1.0
                det3d.bbox.size.x = diameter
                det3d.bbox.size.y = diameter
                det3d.bbox.size.z = diameter
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = name
                hyp.hypothesis.score = score
                hyp.pose.pose.position.x = pt_base.point.x
                hyp.pose.pose.position.y = pt_base.point.y
                hyp.pose.pose.position.z = pt_base.point.z
                hyp.pose.pose.orientation.w = 1.0
                det3d.results.append(hyp)
                det3d_array.detections.append(det3d)

            if self._marker_pub is not None:
                marker = Marker()
                marker.header.frame_id = self._base_frame
                marker.header.stamp = rgb_msg.header.stamp
                marker.ns = self._marker_ns
                marker.id = published
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = pt_base.point.x
                marker.pose.position.y = pt_base.point.y
                marker.pose.position.z = pt_base.point.z
                marker.pose.orientation.w = 1.0
                marker.scale.x = diameter
                marker.scale.y = diameter
                marker.scale.z = diameter
                marker.color.r = 0.1
                marker.color.g = 1.0
                marker.color.b = 0.1
                marker.color.a = float(self._marker_alpha)
                marker.lifetime = Duration(seconds=self._marker_lifetime).to_msg()
                markers.markers.append(marker)

            child = self._tf_frame if single else f"{self._tf_frame}_{published}"
            t = TransformStamped()
            t.header.stamp = rgb_msg.header.stamp
            t.header.frame_id = self._base_frame
            t.child_frame_id = child
            t.transform.translation.x = pt_base.point.x
            t.transform.translation.y = pt_base.point.y
            t.transform.translation.z = pt_base.point.z
            t.transform.rotation.w = 1.0
            transforms.append(t)

            published += 1

        if self._marker_pub is not None:
            # clear spheres left over from a frame that saw more balls.
            for stale_id in range(published, self._prev_marker_count):
                gone = Marker()
                gone.header.frame_id = self._base_frame
                gone.header.stamp = rgb_msg.header.stamp
                gone.ns = self._marker_ns
                gone.id = stale_id
                gone.action = Marker.DELETE
                markers.markers.append(gone)
            self._prev_marker_count = published
            self._marker_pub.publish(markers)

        if self._det3d_pub is not None:
            self._det3d_pub.publish(det3d_array)

        if transforms:
            self._tf_broadcaster.sendTransform(transforms)

        if self._annotated_pub is not None:
            annotated_msg = self._bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            annotated_msg.header = rgb_msg.header
            self._annotated_pub.publish(annotated_msg)

    @staticmethod
    def _draw_label(image, text: str, x: float, y: float) -> None:
        cv2.putText(
            image, text, (int(x), max(0, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    def _transform_point(self, point: PointStamped, header) -> Optional[PointStamped]:
        # The eye-in-hand camera moves with the arm, so the lookup at the
        # exact capture time is the correct one. In sim the image stamp often
        # lands a few ms ahead of the last TF broadcast ("extrapolation into
        # the future") - fall back to the latest available transform, which
        # is close enough while the arm is near-stationary.
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
                    f"could not transform ball point into '{self._base_frame}': {e}"
                )
                return None
        self.get_logger().warning(
            f"could not transform ball point into '{self._base_frame}': "
            f"no usable TF at the image stamp or latest"
        )
        return None


def main() -> None:
    rclpy.init()
    node = BallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
