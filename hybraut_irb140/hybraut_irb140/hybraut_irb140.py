#!/usr/bin/env python3
"""
hybraut_irb140.py

ROS2 node that drives the ABB IRB140 arm through a fixed sequence of
joint-space waypoints using the `hybrid_automaton` framework, closed over
the real `arm_controller` (a ros2_control JointTrajectoryController).

Each waypoint is a discrete automaton state:
  - on entry, it sends a FollowJointTrajectory goal for that waypoint to
    `arm_controller` via its action interface
  - a guard watches live `/joint_states` feedback (injected into the
    automaton's continuous state via a `continuous_state_provider`) and
    fires the transition to the next waypoint once the arm has actually
    arrived, within `joint_tolerance`

The sequence loops forever: the last waypoint transitions back to the
first.
"""

import asyncio
import threading
from typing import Any, Callable, List, Optional

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from hybrid_automaton import Automaton, ContinuousState, RunResult
from hybrid_automaton.definition import State, Transition, guard, invariant


JOINT_NAMES: List[str] = [
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
]

# 8 distinct joint-space goals, transcribed (de-duplicated) from
# abb_irb140_description/state_goals/joint_state_goals.txt.
WAYPOINTS: List[np.ndarray] = [
    np.array([-2.050208, 0.494137, 0.617655, 2.196606, -0.012148, -0.505446]),
    np.array([1.372251, 0.668234, 0.039287, -2.482088, 1.794063, 3.933264]),
    np.array([-1.943394, 0.690225, -0.497412, 1.852492, 1.778983, 1.178443]),
    np.array([1.324499, -1.185257, -1.007711, -3.205914, 1.242398, 5.779120]),
    np.array([2.127492, -1.681526, 0.277965, 2.424852, 1.560747, -5.917350]),
    np.array([-1.288684, 1.789052, -0.966477, 2.318756, 0.067440, 3.702882]),
    np.array([-2.968812, -1.262960, -0.524683, -1.819686, -0.837341, 5.220616]),
    np.array([-0.174045, -0.520390, 0.784773, -1.171244, -0.773671, 0.516616]),
]


class HybrautIRB140(Node, Automaton):
    """
    ros2 node + hybrid automaton that cycles the IRB140 through
    `WAYPOINTS` in order, commanding `arm_controller` and confirming
    arrival from real `/joint_states` feedback.
    """

    NAME = "HybrautIRB140"
    VERSION = "v0.1.0"

    ACTION_NAME = "/arm_controller/follow_joint_trajectory"

    def __init__(self) -> None:
        # Node and Automaton are unrelated base classes (Automaton doesn't
        # chain to super().__init__()), so both constructors are called
        # explicitly rather than through one cooperative super().__init__().
        Node.__init__(self, "hybraut_irb140", namespace="hybraut")

        self.declare_parameter("joint_tolerance", 0.02)
        self.declare_parameter("segment_duration_sec", 3.0)
        self.declare_parameter("control_loop_hz", 20.0)
        self.declare_parameter("feedback_provision_hz", 30.0)

        self._tolerance: float = self.get_parameter("joint_tolerance").value
        self._segment_duration_sec: float = self.get_parameter("segment_duration_sec").value
        self._control_loop_hz: float = self.get_parameter("control_loop_hz").value
        self._feedback_provision_hz: float = self.get_parameter("feedback_provision_hz").value

        self._latest_positions: Optional[np.ndarray] = None
        self._have_joint_state: bool = False

        # last-published discrete state name, used to build the transition
        # string in _publish_automaton_state (mirrors hybraut_nav_tactical's
        # tactical_node._last_automaton_state_name).
        self._last_automaton_state_name: Optional[str] = None

        cb_group = ReentrantCallbackGroup()

        self._joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_cb,
            qos_profile_system_default,
            callback_group=cb_group,
        )

        self._automaton_state_pub = self.create_publisher(
            String,
            "/hybraut/hybraut_irb140/automaton_state",
            qos_profile_system_default,
            callback_group=cb_group,
        )
        self._automaton_transition_event_pub = self.create_publisher(
            String,
            "/hybraut/hybraut_irb140/automaton_transition_event",
            qos_profile_system_default,
            callback_group=cb_group,
        )

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.ACTION_NAME,
            callback_group=cb_group,
        )

        self.get_logger().info(f"waiting for action server '{self.ACTION_NAME}'...")
        self._action_client.wait_for_server()
        self.get_logger().info(f"action server '{self.ACTION_NAME}' is up")

        self._states: List[State] = self._build_states()

        Automaton.__init__(
            self,
            name=HybrautIRB140.NAME,
            version=HybrautIRB140.VERSION,
            states=self._states,
            configuration={
                "joint_tolerance": self._tolerance,
                "segment_duration_sec": self._segment_duration_sec,
                "num_waypoints": len(WAYPOINTS),
            },
            on_entry=self._on_automaton_entry,
            on_exit=self._on_automaton_exit,
        )

    # === automaton construction =============================================

    def _build_states(self) -> List[State]:
        """Build one State per waypoint, wired into a forever-looping cycle."""
        n = len(WAYPOINTS)
        states: List[State] = [
            State(
                name=f"waypoint_{i}",
                initial=(i == 0),
                invariants=[self._make_always_true_invariant(i)],
            )
            for i in range(n)
        ]

        for i, target in enumerate(WAYPOINTS):
            states[i].on_enter = self._make_on_enter(states[i].name, i, target)
            states[i].add_transition(
                Transition(
                    name=f"reached_waypoint_{i}",
                    to_state=states[(i + 1) % n],
                    guards=[self._make_arrival_guard(i, target)],
                )
            )

        return states

    def _make_arrival_guard(self, index: int, target: np.ndarray) -> Callable[[Any], bool]:
        def _arrived(ctx) -> bool:
            current = ctx.continuous_state.latest()
            return bool(np.all(np.abs(current - target) < self._tolerance))

        return guard(_arrived, name=f"arrived_at_waypoint_{index}")

    def _make_always_true_invariant(self, index: int) -> Callable[[Any], bool]:
        def _holds(ctx) -> bool:
            return True

        return invariant(_holds, name=f"still_moving_{index}")

    def _make_on_enter(self, state_name: str, index: int, target: np.ndarray) -> Callable[[], None]:
        def _on_enter() -> None:
            self.get_logger().info(
                f"[{HybrautIRB140.NAME}] entering waypoint {index}, "
                f"sending goal to arm_controller"
            )
            self._publish_automaton_state(state_name)
            self._send_trajectory_goal(target)

        return _on_enter

    def _publish_automaton_state(self, state_name: str) -> None:
        """Edge-triggered (called exactly once per transition, from the
        entered state's on_enter) publish of the automaton's discrete state
        and the transition that produced it - same metadata/topic shape as
        hybraut_nav_tactical's tactical_node._publish_automaton_state_if_changed,
        just event-driven here since this node owns every State/on_enter
        itself rather than needing to poll a third-party automaton."""
        transition_msg = f"'{self._last_automaton_state_name}' -> '{state_name}'"
        self.get_logger().info(f"automaton transition: {transition_msg}")
        self._automaton_state_pub.publish(String(data=state_name))
        self._automaton_transition_event_pub.publish(String(data=transition_msg))
        self._last_automaton_state_name = state_name

    def _on_automaton_entry(self) -> None:
        self.get_logger().info(f">>> Starting {HybrautIRB140.NAME}")
        # the runtime only calls on_enter for states entered *via* a
        # transition, so the initial state's goal has to be kicked off here.
        self._states[0].on_enter()

    def _on_automaton_exit(self) -> None:
        self.get_logger().info(f">>> Exiting {HybrautIRB140.NAME}")

    # === ROS <-> automaton plumbing =========================================

    def _joint_state_cb(self, msg: JointState) -> None:
        name_to_position = dict(zip(msg.name, msg.position))
        try:
            positions = np.array(
                [name_to_position[joint] for joint in JOINT_NAMES], dtype=float
            )
        except KeyError:
            return  # message doesn't (yet) carry all of our joints
        self._latest_positions = positions
        self._have_joint_state = True

    def _provide_joint_feedback(self, ctx) -> np.ndarray:
        """continuous_state_provider: injects real /joint_states feedback."""
        return self._latest_positions

    def _send_trajectory_goal(self, target: np.ndarray) -> None:
        sec = int(self._segment_duration_sec)
        nanosec = int(round((self._segment_duration_sec - sec) * 1e9))

        point = JointTrajectoryPoint()
        point.positions = target.tolist()
        point.time_from_start = Duration(sec=sec, nanosec=nanosec)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = JOINT_NAMES
        goal_msg.trajectory.points = [point]

        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("arm_controller rejected the trajectory goal")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future) -> None:
        result = future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().warning(
                f"trajectory goal finished with error_code={result.error_code}: "
                f"{result.error_string!r}"
            )


def main() -> None:
    rclpy.init()

    node = HybrautIRB140()

    node.get_logger().info("waiting for initial /joint_states feedback...")
    while rclpy.ok() and not node._have_joint_state:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.get_logger().info("received initial /joint_states feedback, activating automaton")

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    async def run_automaton() -> RunResult:
        return await node.activate(
            initial_continuous_state=ContinuousState(
                name="joint_positions",
                x0=node._latest_positions,
                x_labels=JOINT_NAMES,
            ),
            enable_real_time_mode=True,
            enable_self_integration=False,
            delta_time=1.0 / node._control_loop_hz,
            continuous_state_provider=node._provide_joint_feedback,
            continuous_state_provision_rate=node._feedback_provision_hz,
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
