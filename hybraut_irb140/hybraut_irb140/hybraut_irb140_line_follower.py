#!/usr/bin/env python3
"""
hybraut_irb140_line_follower.py

ROS2 node + hybrid automaton that visually servos the IRB140's
end-effector-mounted camera along a black line, with inverse kinematics
solved *inside* the automaton itself: `continuous_dynamics` (the flow hook)
converts a vision-derived Cartesian target into joint velocities via the
manipulator Jacobian each evaluation step, and the framework's own
self-integration turns that into the streamed reference joint trajectory.

Perception is a separate node (`line_detector.py`, mirroring
hybraut_nav_risk/risk_envelope_node.py -> hybraut_nav_tactical/tactical_node.py's
split) - this node just subscribes to its output and pushes it into
auxiliary state, per that same established pattern.

Two states, cycling with the line rather than through a fixed sequence:
    SEARCHING - line not visible, holds position
    TRACKING  - line visible, flow computes qdot = J^+ * twist toward it

Deliberately has *no* continuous_state_provider re-seeding real /joint_states
into the automaton's reference mid-run: tactical_node.py's own
_continuous_state_provider docstring documents exactly the failure mode that
would cause here - re-injecting real (barely-moving) feedback every cycle at
or above the flow's own update rate stomps the flow's one-step correction
before it can accumulate into actual motion. arm_controller's own JTC
position loop is what keeps the *real* arm honest against the reference this
node streams to it, the same way tactical_node's separate immediate-layer
controller does against its own reference.
"""

import asyncio
import threading
from typing import Any, Callable, List, Optional

import numpy as np
import PyKDL

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, QoSProfile, DurabilityPolicy, ReliabilityPolicy

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hybrid_automaton import Automaton, AuxiliaryState, ContinuousState, RunResult
from hybrid_automaton.definition import State, Transition, continuous_dynamics, guard, invariant

from hybraut_irb140.kdl_chain import build_chain


JOINT_NAMES: List[str] = [
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
]

BASE_LINK = "base_link"
TIP_LINK = "tool0"

# latched QoS matching robot_state_publisher's /robot_description publisher
# (transient_local) - confirmed against the live sim this session; a plain
# depth-N subscription misses it entirely if this node starts after
# robot_state_publisher already published it once at its own startup.
_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


class HybrautIRB140LineFollower(Node, Automaton):
    """ros2 node + hybrid automaton that visually servos the IRB140 along a
    black line, solving IK (Jacobian pseudo-inverse) inside the automaton's
    own TRACKING-state flow."""

    NAME = "HybrautIRB140LineFollower"
    VERSION = "v0.1.0"

    ARM_TRAJECTORY_TOPIC = "/arm_controller/joint_trajectory"

    def __init__(self) -> None:
        Node.__init__(self, "hybraut_irb140_line_follower", namespace="hybraut")

        self.declare_parameter("forward_speed", 0.02)       # m/s along the line's tangent
        self.declare_parameter("lateral_gain", 1.5)          # 1/s, P-gain toward line_target
        self.declare_parameter("damping", 0.05)               # damped-least-squares IK damping
        self.declare_parameter("control_loop_hz", 20.0)
        self.declare_parameter("stream_lead_factor", 1.5)     # time_from_start = factor / control_loop_hz

        self._forward_speed: float = self.get_parameter("forward_speed").value
        self._lateral_gain: float = self.get_parameter("lateral_gain").value
        self._damping: float = self.get_parameter("damping").value
        self._control_loop_hz: float = self.get_parameter("control_loop_hz").value
        self._stream_lead_factor: float = self.get_parameter("stream_lead_factor").value

        self._urdf_xml: Optional[str] = None
        self._latest_joint_positions: Optional[np.ndarray] = None
        self._have_joint_state: bool = False
        self._last_automaton_state_name: Optional[str] = None

        cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            String, "/robot_description", self._robot_description_cb, _LATCHED_QOS,
            callback_group=cb_group,
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, qos_profile_system_default,
            callback_group=cb_group,
        )

        self.get_logger().info("waiting for /robot_description...")
        while rclpy.ok() and self._urdf_xml is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info("waiting for initial /joint_states...")
        while rclpy.ok() and not self._have_joint_state:
            rclpy.spin_once(self, timeout_sec=0.5)

        self._chain = build_chain(self._urdf_xml, BASE_LINK, TIP_LINK)
        self._fk_solver = PyKDL.ChainFkSolverPos_recursive(self._chain)
        self._jac_solver = PyKDL.ChainJntToJacSolver(self._chain)

        self._line_target_aux = AuxiliaryState(name="line_target", aux0=np.zeros(4))
        self._line_visible_aux = AuxiliaryState(name="line_visible", aux0=np.array([0.0]))

        self.create_subscription(
            PoseStamped, "/hybraut/hybraut_irb140/line_target", self._line_target_cb,
            qos_profile_system_default, callback_group=cb_group,
        )
        self.create_subscription(
            Bool, "/hybraut/hybraut_irb140/line_visible", self._line_visible_cb,
            qos_profile_system_default, callback_group=cb_group,
        )

        self._automaton_state_pub = self.create_publisher(
            String, "/hybraut/hybraut_irb140_line_follower/automaton_state",
            qos_profile_system_default, callback_group=cb_group,
        )
        self._automaton_transition_event_pub = self.create_publisher(
            String, "/hybraut/hybraut_irb140_line_follower/automaton_transition_event",
            qos_profile_system_default, callback_group=cb_group,
        )
        self._trajectory_pub = self.create_publisher(
            JointTrajectory, self.ARM_TRAJECTORY_TOPIC,
            qos_profile_system_default, callback_group=cb_group,
        )

        self._states: List[State] = self._build_states()

        Automaton.__init__(
            self,
            name=HybrautIRB140LineFollower.NAME,
            version=HybrautIRB140LineFollower.VERSION,
            states=self._states,
            configuration={
                "forward_speed": self._forward_speed,
                "lateral_gain": self._lateral_gain,
                "damping": self._damping,
            },
            on_entry=self._on_automaton_entry,
            on_exit=self._on_automaton_exit,
        )

        self._stream_timer = self.create_timer(
            1.0 / self._control_loop_hz, self._stream_reference_cb, callback_group=cb_group,
        )

    # === automaton construction =============================================

    def _build_states(self) -> List[State]:
        searching = State(
            name="SEARCHING",
            initial=True,
            flow=continuous_dynamics(self._searching_flow, name="hold_position"),
            invariants=[invariant(lambda ctx: True, name="searching_holds")],
        )
        tracking = State(
            name="TRACKING",
            flow=continuous_dynamics(self._tracking_flow, name="visual_servo_ik"),
            invariants=[invariant(lambda ctx: True, name="tracking_holds")],
        )

        searching.on_enter = self._make_state_publisher("SEARCHING")
        tracking.on_enter = self._make_state_publisher("TRACKING")

        searching.add_transition(Transition(
            name="line_found", to_state=tracking,
            guards=[guard(self._line_visible_guard, name="line_visible")],
        ))
        tracking.add_transition(Transition(
            name="line_lost", to_state=searching,
            guards=[guard(self._line_lost_guard, name="line_not_visible")],
        ))

        return [searching, tracking]

    def _make_state_publisher(self, state_name: str) -> Callable[[], None]:
        def _on_enter() -> None:
            self._publish_automaton_state(state_name)
        return _on_enter

    def _publish_automaton_state(self, state_name: str) -> None:
        """Same edge-triggered discrete-state/transition-event metadata
        publish added to the waypoint node (hybraut_irb140.py) - kept
        consistent across both nodes in this package."""
        transition_msg = f"'{self._last_automaton_state_name}' -> '{state_name}'"
        self.get_logger().info(f"automaton transition: {transition_msg}")
        self._automaton_state_pub.publish(String(data=state_name))
        self._automaton_transition_event_pub.publish(String(data=transition_msg))
        self._last_automaton_state_name = state_name

    def _on_automaton_entry(self) -> None:
        self.get_logger().info(f">>> Starting {HybrautIRB140LineFollower.NAME}")
        self._states[0].on_enter()  # runtime never calls on_enter for the initial state itself

    def _on_automaton_exit(self) -> None:
        self.get_logger().info(f">>> Exiting {HybrautIRB140LineFollower.NAME}")

    # === guards ==============================================================

    def _line_visible_guard(self, ctx) -> bool:
        return bool(ctx.auxiliary_states["line_visible"].latest()[0])

    def _line_lost_guard(self, ctx) -> bool:
        return not self._line_visible_guard(ctx)

    # === flows (IK lives here) ==============================================

    def _searching_flow(self, ctx) -> np.ndarray:
        return np.zeros(len(JOINT_NAMES))

    def _tracking_flow(self, ctx) -> np.ndarray:
        """The IK: builds a desired 6D Cartesian twist toward the detected
        line (constant forward speed along its tangent + proportional
        lateral/height correction, orientation held), then solves for joint
        velocities via a damped-least-squares Jacobian pseudo-inverse -
        damped rather than a raw pinv because the arm can pass through
        near-singular configurations while tracking."""
        q = ctx.continuous_state.latest()
        jnt_q = PyKDL.JntArray(len(JOINT_NAMES))
        for i, val in enumerate(q):
            jnt_q[i] = float(val)

        current_frame = PyKDL.Frame()
        self._fk_solver.JntToCart(jnt_q, current_frame)
        current_pos = np.array([current_frame.p.x(), current_frame.p.y(), current_frame.p.z()])

        target = ctx.auxiliary_states["line_target"].latest()
        target_pos = target[:3]
        yaw = float(target[3])
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])

        linear_velocity = self._forward_speed * forward + self._lateral_gain * (target_pos - current_pos)
        angular_velocity = np.zeros(3)  # orientation held (camera kept level) - not error-corrected yet

        twist = np.concatenate([linear_velocity, angular_velocity])

        jacobian = PyKDL.Jacobian(len(JOINT_NAMES))
        self._jac_solver.JntToJac(jnt_q, jacobian)
        J = np.array([[jacobian[r, c] for c in range(len(JOINT_NAMES))] for r in range(6)])

        # damped least squares: qdot = J^T (J J^T + damping^2 I)^-1 twist
        JJt = J @ J.T + (self._damping ** 2) * np.eye(6)
        qdot = J.T @ np.linalg.solve(JJt, twist)
        return qdot

    # === ROS <-> automaton plumbing =========================================

    def _robot_description_cb(self, msg: String) -> None:
        self._urdf_xml = msg.data

    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_position = dict(zip(msg.name, msg.position))
        try:
            positions = np.array([name_to_position[j] for j in JOINT_NAMES], dtype=float)
        except KeyError:
            return
        self._latest_joint_positions = positions
        self._have_joint_state = True

    def _line_target_cb(self, msg: PoseStamped) -> None:
        yaw = 2.0 * np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)
        self._line_target_aux.add(np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, yaw,
        ]))

    def _line_visible_cb(self, msg: Bool) -> None:
        self._line_visible_aux.add(np.array([1.0 if msg.data else 0.0]))

    def _stream_reference_cb(self) -> None:
        """Streams the automaton's current (self-integrated) reference joint
        position to arm_controller as a single-point JointTrajectory on its
        plain topic - not the FollowJointTrajectory action the waypoint node
        uses, since visual servoing needs continuously updated setpoints
        rather than one-shot goals."""
        try:
            reference = self._runtime._ctx.continuous_state.latest()
        except AttributeError:
            return  # automaton not active yet (or just stopped)

        lead_sec = self._stream_lead_factor / self._control_loop_hz
        sec = int(lead_sec)
        nanosec = int(round((lead_sec - sec) * 1e9))

        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in reference]
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)

        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES
        msg.points = [point]
        self._trajectory_pub.publish(msg)


def main() -> None:
    rclpy.init()

    node = HybrautIRB140LineFollower()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    async def run_automaton() -> RunResult:
        return await node.activate(
            initial_continuous_state=ContinuousState(
                name="joint_reference",
                x0=node._latest_joint_positions,
                x_labels=JOINT_NAMES,
            ),
            initial_auxiliary_states=[node._line_target_aux, node._line_visible_aux],
            enable_real_time_mode=True,
            enable_self_integration=True,
            delta_time=1.0 / node._control_loop_hz,
        )

    try:
        result = asyncio.run(run_automaton())
        node.get_logger().info(str(result))
    except KeyboardInterrupt:
        node.get_logger().info("interrupted, deactivating automaton")
        node.deactivate()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
