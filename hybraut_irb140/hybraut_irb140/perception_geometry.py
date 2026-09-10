#!/usr/bin/env python3
"""
perception_geometry.py

Small pure helpers shared by the eye-in-hand perception nodes
(`line_detector.py`, `lego_detector.py`): depth-image unit conversion and
pinhole back-projection of a pixel to a 3D point in the camera's optical
frame. No `rclpy.node.Node` dependency, so they stay unit-testable and
reusable by any future perception node.
"""

from typing import Optional

import numpy as np

from image_geometry import PinholeCameraModel
from geometry_msgs.msg import PointStamped


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """Normalise a raw depth image to float32 metres.

    RealSense-style `16UC1` depth is millimetres; anything else (e.g. the
    Gazebo `32FC1` depth image) is already metres and only needs a dtype
    cast.
    """
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 1000.0
    return depth.astype(np.float32)


def backproject_pixel(
    u: float,
    v: float,
    depth: np.ndarray,
    camera_model: PinholeCameraModel,
    window: int = 1,
) -> Optional[PointStamped]:
    """Back-project pixel ``(u, v)`` through ``depth`` to a 3D point.

    Depth is sampled over a ``(2 * window + 1)`` square median around the
    pixel to suppress single-pixel noise / holes. Returns the point in the
    camera's own optical frame (``camera_model.tfFrame()``), or ``None`` if
    the pixel is outside the image or no valid depth sample is available.
    """
    h, w = depth.shape[:2]
    iu, iv = int(round(u)), int(round(v))
    if not (0 <= iu < w and 0 <= iv < h):
        return None

    r = max(0, int(window))
    patch = depth[max(0, iv - r):iv + r + 1, max(0, iu - r):iu + r + 1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    z = float(np.median(valid))

    fx, fy = camera_model.fx(), camera_model.fy()
    cx0, cy0 = camera_model.cx(), camera_model.cy()

    point = PointStamped()
    point.header.frame_id = camera_model.tfFrame()
    point.point.x = (u - cx0) * z / fx
    point.point.y = (v - cy0) * z / fy
    point.point.z = z
    return point
