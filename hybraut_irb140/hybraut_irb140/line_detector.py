#!/usr/bin/env python3
"""
line_detector.py

Perception node: finds the black line drawn on the sheet of paper on the
table from the end-effector-mounted RGB + depth camera, and republishes its
3D position + tangent direction in `base_link` as a plain topic for
`hybraut_irb140_line_follower.py` to consume - mirrors the
hybraut_nav_risk/risk_envelope_node.py -> hybraut_nav_tactical/tactical_node.py
split (a standalone perception node publishing plain topics that the
automaton node subscribes into its auxiliary state).

Camera topic names default to a RealSense-D405-style layout but are all ROS
parameters, since the camera mount is being added to the URDF separately.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from rclpy.time import Time
from message_filters import ApproximateTimeSynchronizer, Subscriber

from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401 - registers PointStamped transform support

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Bool

from hybraut_irb140.perception_geometry import backproject_pixel, depth_to_meters


class LineDetector(Node):

    def __init__(self) -> None:
        super().__init__("line_detector", namespace="hybraut")

        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("line_intensity_threshold", 60)
        self.declare_parameter("min_contour_area_px", 50.0)
        self.declare_parameter("tangent_step_px", 10.0)
        self.declare_parameter("tf_timeout", 0.1)

        self._base_frame: str = self.get_parameter("base_frame").value
        self._threshold: int = self.get_parameter("line_intensity_threshold").value
        self._min_area: float = self.get_parameter("min_contour_area_px").value
        self._tangent_step_px: float = self.get_parameter("tangent_step_px").value
        self._tf_timeout: float = self.get_parameter("tf_timeout").value

        self._bridge = CvBridge()
        self._camera_model = PinholeCameraModel()
        self._have_camera_info = False

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._line_target_pub = self.create_publisher(
            PoseStamped, "/hybraut/hybraut_irb140/line_target", qos_profile_system_default
        )
        self._line_visible_pub = self.create_publisher(
            Bool, "/hybraut/hybraut_irb140/line_visible", qos_profile_system_default
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

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        self._camera_model.fromCameraInfo(msg)
        self._have_camera_info = True

    def _image_cb(self, rgb_msg: Image, depth_msg: Image) -> None:
        if not self._have_camera_info:
            return

        bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth = depth_to_meters(self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))

        found = self._find_line(bgr)
        if found is None:
            self._line_visible_pub.publish(Bool(data=False))
            return

        (cx, cy), (vx, vy) = found

        point_cam = backproject_pixel(cx, cy, depth, self._camera_model)
        if point_cam is None:
            self._line_visible_pub.publish(Bool(data=False))
            return

        # second point a short pixel-step along the fitted line direction,
        # reusing the same depth sample (the paper is ~flat, so depth barely
        # changes over a few pixels) - just to get a tangent direction.
        tangent_point_cam = backproject_pixel(
            cx + vx * self._tangent_step_px, cy + vy * self._tangent_step_px, depth,
            self._camera_model,
        )

        stamp = rgb_msg.header
        target = self._transform_point(point_cam, stamp)
        if target is None:
            self._line_visible_pub.publish(Bool(data=False))
            return

        yaw = 0.0
        if tangent_point_cam is not None:
            tangent_target = self._transform_point(tangent_point_cam, stamp)
            if tangent_target is not None:
                dx = tangent_target.point.x - target.point.x
                dy = tangent_target.point.y - target.point.y
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    yaw = math.atan2(dy, dx)

        pose = PoseStamped()
        pose.header.frame_id = self._base_frame
        pose.header.stamp = stamp.stamp
        pose.pose.position.x = target.point.x
        pose.pose.position.y = target.point.y
        pose.pose.position.z = target.point.z
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._line_target_pub.publish(pose)
        self._line_visible_pub.publish(Bool(data=True))

    """ === image processing === """

    def _find_line(self, bgr: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Returns ((centroid_x, centroid_y), (tangent_x, tangent_y)) in
        pixel coordinates for the largest dark contour, or None if nothing
        line-like is in view."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self._threshold, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self._min_area:
            return None

        moments = cv2.moments(largest)
        if abs(moments["m00"]) < 1e-6:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

        vx, vy, _, _ = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return (cx, cy), (float(vx), float(vy))

    def _transform_point(self, point: PointStamped, header) -> Optional[PointStamped]:
        # eye-in-hand camera moves with the arm, so the stamped lookup is the
        # correct one; in sim the image stamp often lands a few ms ahead of
        # the last TF broadcast, so fall back to the latest transform.
        for stamp in (header.stamp, Time().to_msg()):
            point.header.stamp = stamp
            try:
                return self._tf_buffer.transform(
                    point, self._base_frame, timeout=Duration(seconds=self._tf_timeout)
                )
            except tf2_ros.ExtrapolationException:
                continue
            except tf2_ros.TransformException as e:
                self.get_logger().warning(
                    f"could not transform line point into '{self._base_frame}': {e}"
                )
                return None
        self.get_logger().warning(
            f"could not transform line point into '{self._base_frame}': "
            f"no usable TF at the image stamp or latest"
        )
        return None


def main() -> None:
    rclpy.init()
    node = LineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
