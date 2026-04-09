from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
import numpy as np


class DaPolicy(Policy):
    """
    Minimal starter policy:
    1. Waits briefly for observations
    2. Logs observation info cleanly
    3. Moves to a fixed pre-grasp pose
    4. Holds position for a moment
    5. Returns True so the task loop can continue cleanly

    This is intentionally simple. It does NOT grasp or insert yet.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("DaPolicy.__init__()")

        # Fixed pre-grasp pose in base_link.
        # Start conservative; adjust after you see the robot in sim.
        self.pregrasp_pose = Pose(
            position=Point(
                x=-0.40,
                y=0.45,
                z=0.22,
            ),
            # Same "look-down" orientation style used by WaveArm.
            orientation=Quaternion(
                x=1.0,
                y=0.0,
                z=0.0,
                w=0.0,
            ),
        )

    def _log_observation_summary(self, observation) -> None:
        """Log a few useful fields without dumping huge message payloads."""
        if observation is None:
            self.get_logger().info("Observation: None")
            return

        try:
            stamp = observation.center_image.header.stamp
            t = stamp.sec + stamp.nanosec / 1e9
            self.get_logger().info(f"Observation time: {t:.3f}s")
        except Exception as ex:
            self.get_logger().info(f"Could not read center_image timestamp: {ex}")

        # These checks are defensive because message layouts may evolve.
        for attr in [
            "center_image",
            "left_image",
            "right_image",
            "joint_state",
            "wrench",
        ]:
            has_attr = hasattr(observation, attr)
            self.get_logger().info(f"observation.{attr}: {'yes' if has_attr else 'no'}")

        # Try a few common image metadata fields if present.
        for img_name in ["center_image", "left_image", "right_image"]:
            if hasattr(observation, img_name):
                img = getattr(observation, img_name)
                pieces = []
                for field in ["height", "width", "encoding"]:
                    if hasattr(img, field):
                        pieces.append(f"{field}={getattr(img, field)}")
                if pieces:
                    self.get_logger().info(f"{img_name}: " + ", ".join(pieces))

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"DaPolicy.insert_cable() enter. Task: {task}")
        send_feedback("DaPolicy: waiting for observation")

        # Wait up to 3 seconds for an observation.
        start_time = self.time_now()
        obs_timeout = Duration(seconds=3.0)
        observation = None

        while (self.time_now() - start_time) < obs_timeout:
            observation = get_observation()
            if observation is not None:
                break
            self.sleep_for(0.1)

        self._log_observation_summary(observation)

        if observation is None:
            self.get_logger().warn("No observation received within timeout.")
            send_feedback("DaPolicy: no observation received")
            return False

        # Move to a fixed pre-grasp pose.
        send_feedback("DaPolicy: moving to fixed pre-grasp pose")
        self.get_logger().info(
            "Moving to pre-grasp pose: "
            f"x={self.pregrasp_pose.position.x:.3f}, "
            f"y={self.pregrasp_pose.position.y:.3f}, "
            f"z={self.pregrasp_pose.position.z:.3f}"
        )

        self.set_pose_target(
            move_robot=move_robot,
            pose=self.pregrasp_pose,
        )

        # Give the controller time to move.
        self.sleep_for(2.0)

        # Log one more observation after the move.
        send_feedback("DaPolicy: checking observation after move")
        observation_after = get_observation()
        self._log_observation_summary(observation_after)

        self.get_logger().info("DaPolicy.insert_cable() exiting.")
        send_feedback("DaPolicy: finished pre-grasp move")
        return True
