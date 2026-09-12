#!/usr/bin/env python3
"""
ball_pick_and_place.py

Final assembly node: consumes ball_detector.py's 3D ball detections
(`/hybraut/hybraut_irb140/ball_detections_3d`, already transformed into
`base_link`), solves inverse kinematics for a pre-grasp/grasp/lift approach
over the detected ball, and drives the arm + pneumatic gripper through a full
pick-and-place -- depositing the ball at the same drop-off joint waypoint
`abb_irb140_description/scripts/irb140_pick_and_place_demo_real.py` uses.

IK: reduced to 3 DOF, not the full 6. The IRB140 has a spherical wrist --
joints 4-6's axes all meet at one point, so joints 1-3 alone determine reach
(position) and 4-6 only ever change orientation. Joints 4-6 are held fixed
at abb_irb140_description's reference "pick" wrist angles
(FIXED_WRIST_JOINTS_RAD) for the whole sequence, and only joints 1-3 solve
for the target position -- a damped Gauss-Newton step using PyKDL's FK
(`ChainFkSolverPos_recursive`) and Jacobian (`ChainJntToJacSolver`) solvers
(chain built with the same `kdl_chain.build_chain()` utility
`hybraut_irb140_line_follower.py` uses), restricted to the position rows and
the first 3 joint columns. This replaced an earlier full-6-DOF, position-only
-weighted `ChainIkSolverPos_LMA` approach: letting the wrist reorient freely
both violated a real hardware requirement (wrist should stay in the same
fixed grasp orientation throughout) and, for lower targets, reliably
converged into an out-of-joint-limit configuration (confirmed against the
real URDF: matching the reference orientation ~8cm lower needs joint_3 past
its limit by ~30 degrees). With the wrist fixed, positioning is a
well-determined 3-equation/3-unknown problem instead, and converges cleanly.

The ball is detected twice, not once: an initial (coarser, farther-away)
detection plans the approach to pre_grasp; after actually arriving there,
a second, closer detection re-solves the grasp/lift descent from a
presumably more accurate reading, rather than committing the whole sequence
to the first detection's possible error.

Every IK solution is sanity-checked (convergence, FK residual, joint
limits, max delta from the seed pose) before it's ever considered for
motion, and by default each planned move (approach, then descent) is printed
and the operator must confirm before any real motion
happens -- see `require_confirmation`.

The move_to()/set_gripper() plumbing (tolerance-based early-exit, background
-task draining before shutdown) is carried over from
abb_irb140_description/scripts/irb140_pick_and_place_demo_real.py, tuned and
debugged against this exact robot + docker/abb-ros1-bridge relay -- reusing
it avoids reintroducing bugs already found and fixed there.
"""

import asyncio
import threading
from typing import List, Optional, Tuple

import numpy as np
import PyKDL

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default, QoSProfile, DurabilityPolicy, ReliabilityPolicy

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from vision_msgs.msg import Detection3DArray

from urdf_parser_py.urdf import URDF

from hybraut_irb140.kdl_chain import build_chain


JOINT_NAMES: List[str] = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

BASE_LINK_DEFAULT = "base_link"
TIP_LINK_DEFAULT = "pneumatic_gripper_base_link"

# Same numeric poses as abb_irb140_description/scripts/irb140_pick_and_place_demo_real.py's
# WAYPOINTS -- duplicated rather than imported, since abb_irb140_description is an
# ament_cmake package (its scripts/ aren't installed as an importable Python module).
# REFERENCE_GRASP_JOINTS_RAD ("pick") supplies FIXED_WRIST_JOINTS_RAD (joints 4-6 -- see
# module docstring: held fixed for the whole sequence, never solved for).
# DROP_OFF_JOINTS_RAD ("move_to_box") and RETURN_JOINTS_RAD ("return_to_base") are used
# as-is, joint-space, exactly like that script. Re-jog and update all three if the
# workspace/mount changes.
REFERENCE_GRASP_JOINTS_RAD = [-0.12439039349555969, 0.9279851317405701, -0.18804606795310974,
                               0.05352947860956192, 0.7330714464187622, -0.1924125701189041]
FIXED_WRIST_JOINTS_RAD = REFERENCE_GRASP_JOINTS_RAD[3:6]  # joint_4, joint_5, joint_6
DROP_OFF_JOINTS_RAD = [-0.38488104939460754, 1.0068862438201904, -1.0835121870040894,
                        0.06004104018211365, 1.5615001916885376, -0.4128890931606293]
RETURN_JOINTS_RAD = [-0.12438856065273285, 0.5640290379524231, -0.4248206317424774,
                      0.03683801367878914, 1.333264946937561, -0.16127046942710876]

# move_to() pacing/tolerance/deadline -- same values validated this session against the
# real ABB IRB140 + docker/abb-ros1-bridge relay (see irb140_pick_and_place_demo_real.py's
# history: the relay's own GOAL_TIME_SLACK must be generous, currently set to 20.0s in
# docker/abb-ros1-bridge/docker-compose.yml, for this GOAL_TIME_TOLERANCE_SEC to hold).
NOMINAL_JOINT_SPEED_RAD_S = 0.4
MIN_MOVE_DURATION_SEC = 1.5
JOINT_TOLERANCE_RAD = 0.02
GOAL_TIME_TOLERANCE_SEC = 10.0

# Reduced (3-DOF, position-only) IK solve -- damped Gauss-Newton over joints 1-3 only.
IK_MAX_ITERS = 200
IK_POSITION_TOL_M = 1e-6
IK_DAMPING = 0.02

# IK solution safety checks -- reject and abort rather than ever send a bad trajectory.
MAX_IK_POSITION_ERROR_M = 0.01   # FK-residual check on the solved joints
MAX_IK_JOINT_DELTA_RAD = 2.0     # vs. the seed -- catches solver pathologies (e.g. elbow flips)
JOINT_LIMIT_MARGIN_RAD = 0.01

# latched QoS matching robot_state_publisher's /robot_description publisher (transient_local)
# -- see hybraut_irb140_line_follower.py, same reasoning.
_LATCHED_QOS = QoSProfile(
    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE,
)


def move_duration_sec(start_positions: list, target_positions: list) -> float:
    """Scale a move's duration to its largest single-joint delta (same helper as
    irb140_pick_and_place_demo_real.py)."""
    max_delta = max(abs(t - s) for s, t in zip(start_positions, target_positions))
    return max(MIN_MOVE_DURATION_SEC, max_delta / NOMINAL_JOINT_SPEED_RAD_S)


def _jnt_array(values: List[float]) -> PyKDL.JntArray:
    q = PyKDL.JntArray(len(values))
    for i, v in enumerate(values):
        q[i] = float(v)
    return q


def _frame_position_error(a: PyKDL.Frame, b: PyKDL.Frame) -> float:
    d = a.p - b.p
    return (d.x() ** 2 + d.y() ** 2 + d.z() ** 2) ** 0.5


class BallPickAndPlaceNode(Node):

    def __init__(self) -> None:
        super().__init__("ball_pick_and_place", namespace="hybraut")

        self.declare_parameter("base_link", BASE_LINK_DEFAULT)
        self.declare_parameter("tip_link", TIP_LINK_DEFAULT)
        self.declare_parameter("ball_detections_topic",
                                "/hybraut/hybraut_irb140/ball_detections_3d")
        self.declare_parameter("pre_grasp_height_m", 0.15)
        self.declare_parameter("grasp_height_m", 0.0)
        self.declare_parameter("lift_height_m", 0.12)
        # Hard floor, base_link Z (m) -- no target (pre_grasp/grasp/lift) is ever
        # allowed below this, independent of what the ball detection or IK say.
        # No safe default exists (it depends entirely on your physical setup), so
        # this is NaN until you set it explicitly -- the node refuses to run
        # rather than guess. Measure it by jogging the gripper down to just touch
        # the table (slowly, via the pendant) and reading the resulting
        # pneumatic_gripper_base_link Z in base_link -- e.g.
        # `ros2 run tf2_ros tf2_echo base_link pneumatic_gripper_base_link`
        # -- then pass e.g. -p min_z_m:=-0.09 (a couple cm above that reading,
        # not exactly at it).
        self.declare_parameter("min_z_m", float("nan"))
        self.declare_parameter("detection_timeout_sec", 15.0)
        self.declare_parameter("max_detection_age_sec", 1.0)
        self.declare_parameter("min_score", 0.3)
        self.declare_parameter("grip_settle_sec", 0.5)
        self.declare_parameter("require_confirmation", True)

        self._base_link: str = self.get_parameter("base_link").value
        self._tip_link: str = self.get_parameter("tip_link").value
        self._pre_grasp_height_m: float = self.get_parameter("pre_grasp_height_m").value
        self._grasp_height_m: float = self.get_parameter("grasp_height_m").value
        self._lift_height_m: float = self.get_parameter("lift_height_m").value
        self._min_z_m: float = self.get_parameter("min_z_m").value
        if self._min_z_m != self._min_z_m:  # NaN != NaN -- unset
            raise RuntimeError(
                "min_z_m is not set. This is a required, no-default safety floor "
                "(base_link Z, metres) that no grasp target is ever allowed below. "
                "Measure your table height in base_link (e.g. jog the gripper down "
                "to just touch it, slowly, via the pendant, then "
                "`ros2 run tf2_ros tf2_echo base_link pneumatic_gripper_base_link`) "
                "and pass it with a couple cm of margin, e.g.: "
                "ros2 run hybraut_irb140 hybraut_irb140_ball_pick_and_place "
                "--ros-args -p min_z_m:=-0.09"
            )
        self._detection_timeout_sec: float = self.get_parameter("detection_timeout_sec").value
        self._max_detection_age_sec: float = self.get_parameter("max_detection_age_sec").value
        self._min_score: float = self.get_parameter("min_score").value
        self._grip_settle_sec: float = self.get_parameter("grip_settle_sec").value
        self._require_confirmation: bool = self.get_parameter("require_confirmation").value

        self._urdf_xml: Optional[str] = None
        self._joint_limits: Optional[List[Tuple[float, float]]] = None
        self._latest_joint_positions: Optional[List[float]] = None
        self._latest_detections: Optional[Detection3DArray] = None
        self._latest_detections_time = None

        self._background_tasks: set = set()

        cb_group = ReentrantCallbackGroup()

        self.create_subscription(
            String, "/robot_description", self._robot_description_cb, _LATCHED_QOS,
            callback_group=cb_group,
        )
        self.create_subscription(
            JointState, "/joint_states", self._joint_state_cb, qos_profile_system_default,
            callback_group=cb_group,
        )
        self.create_subscription(
            Detection3DArray, self.get_parameter("ball_detections_topic").value,
            self._on_detections, qos_profile_system_default, callback_group=cb_group,
        )

        self.follow_joint_trajectory_cli = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory",
            callback_group=cb_group,
        )
        self.gripper_trigger_cli = self.create_client(
            SetBool, "/gripper_trigger", callback_group=cb_group,
        )

        self.get_logger().info("waiting for /robot_description...")
        while rclpy.ok() and self._urdf_xml is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        self.get_logger().info("waiting for initial /joint_states...")
        while rclpy.ok() and self._latest_joint_positions is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        self._build_kinematics()

    # === kinematics setup ====================================================

    def _build_kinematics(self) -> None:
        self._chain = build_chain(self._urdf_xml, self._base_link, self._tip_link)
        self._fk_solver = PyKDL.ChainFkSolverPos_recursive(self._chain)
        self._jac_solver = PyKDL.ChainJntToJacSolver(self._chain)
        self._joint_limits = self._read_joint_limits()
        self.get_logger().info(
            f"kinematics ready: chain '{self._base_link}' -> '{self._tip_link}'; "
            f"joints 4-6 held fixed at {[round(v, 4) for v in FIXED_WRIST_JOINTS_RAD]} rad "
            f"(REFERENCE_GRASP_JOINTS_RAD's wrist angles) -- only joints 1-3 solve for position"
        )

    def _read_joint_limits(self) -> List[Tuple[float, float]]:
        robot = URDF.from_xml_string(self._urdf_xml)
        joints = {j.name: j for j in robot.joints}
        limits = []
        for name in JOINT_NAMES:
            joint = joints[name]
            limits.append((joint.limit.lower, joint.limit.upper))
        return limits

    # === IK ===================================================================

    def _solve_ik_checked(self, name: str, seed: List[float],
                           target_xyz: Tuple[float, float, float]) -> Optional[List[float]]:
        if target_xyz[2] < self._min_z_m:
            self.get_logger().error(
                f"'{name}' target z={target_xyz[2]:.3f} m is below the min_z_m safety "
                f"floor ({self._min_z_m:.3f} m) -- refusing before even attempting IK. "
                f"This does not necessarily mean the ball detection is wrong; it may "
                f"just mean the height offset params (pre_grasp_height_m/grasp_height_m/"
                f"lift_height_m) need adjusting for your setup."
            )
            return None

        # Position-only, 3-DOF: joints 4-6 stay fixed throughout (see module
        # docstring); only joints 1-3 are solved for, via damped Gauss-Newton
        # on the position rows / first-3-columns of the full Jacobian.
        solution = [seed[0], seed[1], seed[2], *FIXED_WRIST_JOINTS_RAD]
        target = np.array(target_xyz)
        converged = False
        for _ in range(IK_MAX_ITERS):
            frame = PyKDL.Frame()
            self._fk_solver.JntToCart(_jnt_array(solution), frame)
            pos = np.array([frame.p.x(), frame.p.y(), frame.p.z()])
            err = target - pos
            if np.linalg.norm(err) < IK_POSITION_TOL_M:
                converged = True
                break
            jacobian = PyKDL.Jacobian(len(solution))
            self._jac_solver.JntToJac(_jnt_array(solution), jacobian)
            J = np.array([[jacobian[r, c] for c in range(3)] for r in range(3)])
            JJt = J @ J.T + (IK_DAMPING ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, err)
            solution[0] += dq[0]
            solution[1] += dq[1]
            solution[2] += dq[2]

        if not converged:
            self.get_logger().error(
                f"IK for '{name}' did not converge within {IK_MAX_ITERS} iterations, "
                f"target={target_xyz}"
            )
            return None

        target_frame = PyKDL.Frame(PyKDL.Rotation.Identity(), PyKDL.Vector(*target_xyz))
        achieved_frame = PyKDL.Frame()
        self._fk_solver.JntToCart(_jnt_array(solution), achieved_frame)
        position_error = _frame_position_error(target_frame, achieved_frame)
        if position_error > MAX_IK_POSITION_ERROR_M:
            self.get_logger().error(
                f"IK for '{name}' residual too large: {position_error:.4f} m "
                f"(max {MAX_IK_POSITION_ERROR_M} m) -- target likely unreachable"
            )
            return None

        for i, (lower, upper) in enumerate(self._joint_limits):
            if not (lower - JOINT_LIMIT_MARGIN_RAD <= solution[i] <= upper + JOINT_LIMIT_MARGIN_RAD):
                self.get_logger().error(
                    f"IK for '{name}' violates joint limit on {JOINT_NAMES[i]}: "
                    f"{solution[i]:.3f} rad not in [{lower:.3f}, {upper:.3f}]"
                )
                return None

        max_delta = max(abs(a - b) for a, b in zip(seed, solution))
        if max_delta > MAX_IK_JOINT_DELTA_RAD:
            self.get_logger().error(
                f"IK for '{name}' jumped {max_delta:.3f} rad from its seed (max "
                f"{MAX_IK_JOINT_DELTA_RAD} rad) -- rejecting as a likely solver pathology "
                f"(e.g. elbow flip) rather than risk an unexpected sweep"
            )
            return None

        return solution

    # === perception ===========================================================

    def _on_detections(self, msg: Detection3DArray) -> None:
        self._latest_detections = msg
        self._latest_detections_time = self.get_clock().now()

    async def _detect_ball(self, context: str):
        """_wait_for_ball() plus logging. `context` labels which stage of the
        sequence this detection is for (e.g. "initial", "refined")."""
        self.get_logger().info(
            f"waiting up to {self._detection_timeout_sec:.0f}s for a {context} ball "
            f"detection (min_score={self._min_score})..."
        )
        detection = await self._wait_for_ball()
        if detection is None:
            self.get_logger().error(f"no ball detected in time ({context}) -- aborting.")
            return None
        hyp = detection.results[0].hypothesis
        p = detection.results[0].pose.pose.position
        self.get_logger().info(
            f"{context} target '{hyp.class_id}' (score={hyp.score:.2f}) at "
            f"{self._base_link}=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
        )
        return detection

    async def _wait_for_ball(self):
        """Waits for a fresh, sufficiently-confident ball detection. Returns the
        best-scoring Detection3D, or None on timeout."""
        deadline = self.get_clock().now() + rclpy.duration.Duration(
            seconds=self._detection_timeout_sec
        )
        while rclpy.ok() and self.get_clock().now() < deadline:
            msg = self._latest_detections
            stamp = self._latest_detections_time
            if msg is not None and stamp is not None and msg.detections:
                age_sec = (self.get_clock().now() - stamp).nanoseconds * 1e-9
                if age_sec <= self._max_detection_age_sec:
                    if msg.header.frame_id and msg.header.frame_id != self._base_link:
                        self.get_logger().warn(
                            f"ball_detections_3d frame_id='{msg.header.frame_id}' != "
                            f"this node's base_link='{self._base_link}' -- IK targets "
                            f"would be wrong; fix the base_frame params to match"
                        )
                    scored = [d for d in msg.detections if d.results]
                    if scored:
                        best = max(scored, key=lambda d: d.results[0].hypothesis.score)
                        if best.results[0].hypothesis.score >= self._min_score:
                            return best
            await asyncio.sleep(0.1)
        return None

    # === joint state / motion (ported from irb140_pick_and_place_demo_real.py) ==

    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_position = dict(zip(msg.name, msg.position))
        try:
            self._latest_joint_positions = [name_to_position[j] for j in JOINT_NAMES]
        except KeyError:
            return  # message doesn't (yet) carry all of our joints

    def _robot_description_cb(self, msg: String) -> None:
        self._urdf_xml = msg.data

    async def _rclpy_future_to_asyncio(self, rclpy_future):
        loop = asyncio.get_event_loop()
        asyncio_future = loop.create_future()

        def done_callback(f):
            try:
                result = f.result()
                loop.call_soon_threadsafe(asyncio_future.set_result, result)
            except Exception as e:
                loop.call_soon_threadsafe(asyncio_future.set_exception, e)

        rclpy_future.add_done_callback(done_callback)
        return await asyncio_future

    def _within_tolerance(self, target_positions: list, tolerance: float) -> bool:
        current = self._latest_joint_positions
        if current is None:
            return False
        return all(abs(c - t) <= tolerance for c, t in zip(current, target_positions))

    async def _wait_until_reached(self, target_positions: list, tolerance: float,
                                   poll_period_sec: float = 0.05, settle_checks: int = 2):
        consecutive = 0
        while consecutive < settle_checks:
            if self._within_tolerance(target_positions, tolerance):
                consecutive += 1
            else:
                consecutive = 0
            await asyncio.sleep(poll_period_sec)

    def feedback_callback(self, feedback):
        self.get_logger().debug(f"Feedback received: {feedback}")

    async def move_to(self, name: str, positions: list, duration_sec: float,
                       tolerance: float = JOINT_TOLERANCE_RAD) -> bool:
        """Send a single-point FollowJointTrajectory goal, then advance as soon
        as /joint_states shows we're within tolerance of the target -- whichever
        comes first between that and the controller/bridge's own result. Ported
        from irb140_pick_and_place_demo_real.py (see that file's history for why
        it's structured this way -- tolerance-gating, not canceling superseded
        goals, and draining background tasks before shutdown)."""
        self.get_logger().info(f"Moving to '{name}' ({duration_sec:.1f}s)...")

        sec = int(duration_sec)
        nanosec = int((duration_sec - sec) * 1e9)

        goal = FollowJointTrajectory.Goal(
            trajectory=JointTrajectory(
                joint_names=JOINT_NAMES,
                points=[JointTrajectoryPoint(
                    positions=positions,
                    time_from_start=DurationMsg(sec=sec, nanosec=nanosec),
                )],
            ),
            goal_time_tolerance=DurationMsg(sec=int(GOAL_TIME_TOLERANCE_SEC), nanosec=0),
        )

        send_goal_future = self.follow_joint_trajectory_cli.send_goal_async(
            goal, feedback_callback=self.feedback_callback
        )
        goal_handle = await self._rclpy_future_to_asyncio(send_goal_future)

        if not goal_handle.accepted:
            self.get_logger().error(f"Goal '{name}' rejected!")
            return False

        result_task = asyncio.ensure_future(
            self._rclpy_future_to_asyncio(goal_handle.get_result_async())
        )
        reached_task = asyncio.ensure_future(self._wait_until_reached(positions, tolerance))

        done, _pending = await asyncio.wait(
            {result_task, reached_task}, return_when=asyncio.FIRST_COMPLETED
        )

        if reached_task in done:
            self.get_logger().info(f"Reached '{name}' (within {tolerance} rad).")
            self._background_tasks.add(result_task)
            result_task.add_done_callback(self._background_tasks.discard)
            result_task.add_done_callback(
                lambda f, name=name: self._log_late_result(name, f)
            )
            return True

        reached_task.cancel()
        result = result_task.result()
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f"Goal '{name}' finished with non-success status: {result.status}"
            )
            return False

        self.get_logger().info(f"Reached '{name}'.")
        return True

    def _log_late_result(self, name: str, result_task: "asyncio.Task"):
        try:
            result = result_task.result()
        except asyncio.CancelledError:
            self.get_logger().debug(f"Goal '{name}' result task canceled (shutdown).")
            return
        except Exception as e:  # noqa: BLE001
            self.get_logger().debug(f"Goal '{name}' result future errored after being superseded: {e}")
            return
        self.get_logger().debug(
            f"Goal '{name}' (already superseded) later resolved with status {result.status}."
        )

    async def set_gripper(self, closed: bool) -> bool:
        action = "Closing" if closed else "Opening"
        self.get_logger().info(f"{action} gripper...")

        request = SetBool.Request(data=closed)
        response_future = self.gripper_trigger_cli.call_async(request)
        response = await self._rclpy_future_to_asyncio(response_future)

        if not response.success:
            self.get_logger().error(f"Gripper trigger failed: {response.message}")
            return False

        self.get_logger().info(f"Gripper trigger succeeded: {response.message}")
        return True

    # === confirmation gate =====================================================

    def _print_plan(self, label: str, ball_xyz, named_joints: dict) -> None:
        def fmt(q):
            return "[" + ", ".join(f"{v:+.3f}" for v in q) + "]"

        print(f"\n=== ball_pick_and_place: {label} plan ===")
        print(f"  ball (base_link):  x={ball_xyz[0]:+.3f}  y={ball_xyz[1]:+.3f}  z={ball_xyz[2]:+.3f}")
        for name, q in named_joints.items():
            print(f"  {name} joints:  {fmt(q)}")
        print("===========================================\n")

    async def _confirm(self, prompt: str) -> bool:
        def _read():
            try:
                return input(prompt).strip().lower()
            except EOFError:
                return "n"
        answer = await asyncio.to_thread(_read)
        return answer in ("", "y", "yes")

    # === main sequence =========================================================

    async def run_pick_and_place(self) -> bool:
        if not await self.set_gripper(closed=False):
            return False

        # --- stage 1: coarse detection plans the approach to pre_grasp -------
        detection = await self._detect_ball("initial")
        if detection is None:
            return False
        p = detection.results[0].pose.pose.position

        current = self._latest_joint_positions
        if current is None:
            self.get_logger().error("no /joint_states yet -- aborting.")
            return False

        pre_grasp_q = self._solve_ik_checked(
            "pre_grasp", current, (p.x, p.y, p.z + self._pre_grasp_height_m)
        )
        if pre_grasp_q is None:
            return False

        if self._require_confirmation:
            self._print_plan("approach", (p.x, p.y, p.z), {"pre_grasp": pre_grasp_q})
            if not await self._confirm("Move to pre-grasp above the ball? [Y/n] "):
                self.get_logger().warn("aborted by operator before any motion.")
                return False

        if not await self.move_to("pre_grasp", pre_grasp_q, move_duration_sec(current, pre_grasp_q)):
            return False

        # --- stage 2: re-detect from pre_grasp (closer, better view) before ---
        # committing to the descent -- corrects for error/bias in the farther-
        # away initial detection, which is what was causing missed grasps.
        refined = await self._detect_ball("refined")
        if refined is None:
            return False
        rp = refined.results[0].pose.pose.position

        grasp_q = self._solve_ik_checked(
            "grasp", pre_grasp_q, (rp.x, rp.y, rp.z + self._grasp_height_m)
        )
        if grasp_q is None:
            return False

        lift_q = self._solve_ik_checked(
            "lift", grasp_q, (rp.x, rp.y, rp.z + self._lift_height_m)
        )
        if lift_q is None:
            return False

        if self._require_confirmation:
            self._print_plan("descent", (rp.x, rp.y, rp.z), {"grasp": grasp_q, "lift": lift_q})
            if not await self._confirm("Descend and grasp? [Y/n] "):
                self.get_logger().warn("aborted by operator before descent.")
                return False

        if not await self.move_to("grasp", grasp_q, move_duration_sec(pre_grasp_q, grasp_q)):
            return False
        if not await self.set_gripper(closed=True):
            return False
        await asyncio.sleep(self._grip_settle_sec)
        if not await self.move_to("lift", lift_q, move_duration_sec(grasp_q, lift_q)):
            return False
        if not await self.move_to("drop_off", DROP_OFF_JOINTS_RAD,
                                   move_duration_sec(lift_q, DROP_OFF_JOINTS_RAD)):
            return False
        if not await self.set_gripper(closed=False):
            return False
        if not await self.move_to("return_home", RETURN_JOINTS_RAD,
                                   move_duration_sec(DROP_OFF_JOINTS_RAD, RETURN_JOINTS_RAD)):
            return False

        return True


async def _async_main():
    rclpy.init()
    node = BallPickAndPlaceNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    if not node.follow_joint_trajectory_cli.wait_for_server(timeout_sec=10.0):
        node.get_logger().error("Action server not available!")
        executor.shutdown()
        rclpy.shutdown()
        return

    if not node.gripper_trigger_cli.wait_for_service(timeout_sec=10.0):
        node.get_logger().error("Gripper trigger service not available!")
        executor.shutdown()
        rclpy.shutdown()
        return

    success = await node.run_pick_and_place()
    if success:
        node.get_logger().info("Ball pick-and-place completed successfully.")
    else:
        node.get_logger().error("Ball pick-and-place failed — see errors above.")

    if node._background_tasks:
        for t in list(node._background_tasks):
            t.cancel()
        await asyncio.gather(*node._background_tasks, return_exceptions=True)

    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


def main() -> None:
    # ros2 run's console_scripts entry point calls main() directly (never
    # runs this file as __main__), so main() itself has to be the sync
    # entry that drives the async work -- same pattern as hybraut_irb140.py
    # and hybraut_irb140_line_follower.py in this package.
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
