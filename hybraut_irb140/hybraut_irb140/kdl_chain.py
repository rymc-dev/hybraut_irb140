#!/usr/bin/env python3
"""
kdl_chain.py

Builds a PyKDL.Chain for a base_link -> tip_link kinematic chain directly
from a URDF XML string, via urdf_parser_py.

`kdl_parser_py` (the usual way to do this) isn't packaged for this ROS2
distro (confirmed: `ros-jazzy-kdl-parser` only ships the C++ library;
`kdl_parser-py` only exists for `ros-rolling`) - so this reimplements the
same URDF-joint -> KDL::Joint/Segment mapping kdl_parser's C++ source uses,
using `urdf_parser_py` (available) for the URDF side and `PyKDL` (available)
for the KDL side.
"""

from typing import List

import PyKDL
from urdf_parser_py.urdf import URDF, Joint as UrdfJoint


def _kdl_frame(xyz, rpy) -> PyKDL.Frame:
    xyz = xyz or [0.0, 0.0, 0.0]
    rpy = rpy or [0.0, 0.0, 0.0]
    return PyKDL.Frame(PyKDL.Rotation.RPY(*rpy), PyKDL.Vector(*xyz))


def _origin_frame(joint: UrdfJoint) -> PyKDL.Frame:
    if joint.origin is None:
        return PyKDL.Frame.Identity()
    return _kdl_frame(joint.origin.xyz, joint.origin.rpy)


def _kdl_joint(joint: UrdfJoint) -> PyKDL.Joint:
    """
    Mirrors kdl_parser's toKdl(urdf::Joint): the KDL::Joint's own origin/axis
    are expressed in the *parent* link frame (i.e. the URDF joint's <origin>
    transform is folded into the joint definition itself, axis rotated into
    the parent frame by that same origin), and the enclosing Segment's fixed
    frame is set to that identical origin transform - so the joint rotates
    about the right axis/point *and* the segment ends up at the right place
    for a purely revolute/fixed chain with no separate post-joint offset.
    """
    origin = _origin_frame(joint)

    if joint.type == "fixed" or joint.axis is None:
        return PyKDL.Joint(joint.name, PyKDL.Joint.Fixed)

    axis_in_parent = origin.M * PyKDL.Vector(*joint.axis)

    if joint.type in ("revolute", "continuous"):
        return PyKDL.Joint(joint.name, origin.p, axis_in_parent, PyKDL.Joint.RotAxis)
    if joint.type == "prismatic":
        return PyKDL.Joint(joint.name, origin.p, axis_in_parent, PyKDL.Joint.TransAxis)

    raise ValueError(f"unsupported joint type '{joint.type}' for joint '{joint.name}'")


def _joint_path(robot: URDF, base_link: str, tip_link: str) -> List[UrdfJoint]:
    """Walks parent<-child joints from tip_link back to base_link, then
    reverses - URDF is a tree, so this is the unique path between them."""
    child_to_joint = {joint.child: joint for joint in robot.joints}

    path: List[UrdfJoint] = []
    link = tip_link
    while link != base_link:
        joint = child_to_joint.get(link)
        if joint is None:
            raise ValueError(
                f"no kinematic path from '{base_link}' to '{tip_link}' "
                f"(no parent joint found for link '{link}')"
            )
        path.append(joint)
        link = joint.parent
    path.reverse()
    return path


def build_chain(urdf_xml: str, base_link: str, tip_link: str) -> PyKDL.Chain:
    """Builds the PyKDL.Chain for the base_link -> tip_link path in urdf_xml."""
    robot = URDF.from_xml_string(urdf_xml)
    chain = PyKDL.Chain()
    for joint in _joint_path(robot, base_link, tip_link):
        chain.addSegment(PyKDL.Segment(joint.child, _kdl_joint(joint), _origin_frame(joint)))
    return chain
