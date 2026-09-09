# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Optional

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.materials import PreviewSurfaceMaterial, RigidBodyMaterial
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from .franka_experimental import FrankaExperimental
from isaacsim.storage.native import get_assets_root_path


class FrankaPickPlace:
    """Simple, direct Franka pick-and-place controller.

    No complex inheritance, no RL concepts, no task wrappers.
    Just straightforward robot control that's easy to understand and modify.
    """

    def __init__(
        self,
        events_dt: Optional[List[float]] = None,
        reset_settle_steps: int = 15,
        fault_open_gripper: bool = False,
        fault_zero_cube_friction: bool = False,
        fault_high_grasp_approach: bool = False,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        """Initialize the FrankaPickPlace controller.

        Sets up initial state variables for the state machine.

        Args:
            events_dt: List of step counts for each phase. If None, uses default values.
            reset_settle_steps: Number of simulation steps to hold the reset state before moving.
            fault_open_gripper: Keep the fingers open during phase 2 instead of grasping.
            fault_zero_cube_friction: Remove friction from the cube collision surface.
            fault_high_grasp_approach: Approach the grasp pose 0.08 m too high.
            log_callback: Optional callback for experiment logging.
        """
        self.cube = None
        self.robot = None
        self.reset_settle_steps = max(0, int(reset_settle_steps))
        self.fault_open_gripper = bool(fault_open_gripper)
        self.fault_zero_cube_friction = bool(fault_zero_cube_friction)
        self.fault_high_grasp_approach = bool(fault_high_grasp_approach)
        if sum(
            (
                self.fault_open_gripper,
                self.fault_zero_cube_friction,
                self.fault_high_grasp_approach,
            )
        ) > 1:
            raise ValueError("controlled fault options are mutually exclusive")
        self.grasp_height_offset_m = 0.08 if self.fault_high_grasp_approach else 0.0
        self._log_callback = log_callback
        self._run_start_cube_position = None
        self._lift_target = None
        self._retract_target = None
        self._reset_settle_steps_remaining = 0
        self._run_step_count = 0
        self._trace: List[Dict[str, Any]] = []
        self._phase_history: List[Dict[str, Any]] = []
        self._max_cube_height = float("-inf")
        self._min_cube_ee_distance_during_lift = float("inf")
        self._min_gripper_finger_sum_during_grasp = float("inf")
        self._grasp_ee_distance_at_command = None
        self._last_observation = None

        # Define step counts for each phase
        self.events_dt = events_dt
        if self.events_dt is None:
            # Phase durations: [(x,y) positioning, approach, grasp, lift, move, release, retract]
            self.events_dt = [
                60,  # Phase 0: Move to x,y position above cube
                40,  # Phase 1: Approach down to cube
                20,  # Phase 2: Close gripper to grasp
                40,  # Phase 3: Lift cube upward
                80,  # Phase 4: Move cube to target location
                20,  # Phase 5: Open gripper to release
                20,  # Phase 6: Move up and away
            ]
        self._event = 0
        self._step = 0

    def _emit(self, message: str) -> None:
        """Emit a human-readable event to stdout and the experiment logger."""
        if self._log_callback is not None:
            self._log_callback(message)
        else:
            print(message, flush=True)

    @staticmethod
    def _as_list(value: Any) -> Optional[List[float]]:
        if value is None:
            return None
        return np.asarray(value, dtype=float).reshape(-1).tolist()

    def _observe(self, label: str = "step") -> Dict[str, Any]:
        """Record state useful for separating motion, contact, and placement errors."""
        if self.cube is None or self.robot is None:
            return {"label": label, "event": self._event, "step": self._step}

        cube_positions, cube_orientations = self.cube.get_world_poses()
        cube_position = cube_positions.numpy()[0]
        cube_orientation = cube_orientations.numpy()[0]
        _, ee_positions, ee_orientations = self.robot.get_current_state()
        ee_position = ee_positions[0]
        ee_orientation = ee_orientations[0]
        dof_positions = self.robot.get_dof_positions().numpy()[0]

        cube_ee_distance = float(np.linalg.norm(cube_position - ee_position))
        self._max_cube_height = max(self._max_cube_height, float(cube_position[2]))
        if self._event >= 3:
            self._min_cube_ee_distance_during_lift = min(
                self._min_cube_ee_distance_during_lift, cube_ee_distance
            )
        if 2 <= self._event <= 4 and dof_positions.shape[0] >= 9:
            self._min_gripper_finger_sum_during_grasp = min(
                self._min_gripper_finger_sum_during_grasp,
                float(dof_positions[7] + dof_positions[8]),
            )

        observation = {
            "label": label,
            "run_step": self._run_step_count,
            "event": self._event,
            "phase": self.phase_name,
            "phase_step": self._step,
            "cube_position": cube_position.tolist(),
            "cube_orientation": cube_orientation.tolist(),
            "end_effector_position": ee_position.tolist(),
            "end_effector_orientation": ee_orientation.tolist(),
            "dof_positions": dof_positions.tolist(),
            "cube_ee_distance": cube_ee_distance,
        }
        self._trace.append(observation)
        self._last_observation = observation
        return observation

    @property
    def phase_name(self) -> str:
        return (
            "move_above_cube",
            "approach_cube",
            "close_gripper",
            "lift_cube",
            "move_to_target",
            "release_cube",
            "retract",
        )[min(self._event, 6)]

    def setup_scene(
        self,
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        """Set up the scene with robot and cube.

        Creates and adds a Franka robot and a dynamic cube to the simulation world.
        Sets default values for positions and sizes if not provided.

        Args:
            cube_initial_position: Initial cube position [x, y, z]. Defaults to [0.5, 0.0, 0.0258].
            cube_initial_orientation: Initial cube orientation as quaternion [w, x, y, z]. Defaults to [1, 0, 0, 0].
            cube_size: Cube dimensions [w, h, d]. Defaults to [0.0515, 0.0515, 0.0515].
            target_position: Target position for placing [x, y, z]. Defaults to [-0.3, -0.3, cube_height/2].
            offset: Additional offset to apply to target position [x, y, z]. Defaults to [0, 0, 0].
        """
        self.cube_initial_position = cube_initial_position
        self.cube_initial_orientation = cube_initial_orientation
        self.target_position = target_position
        self.cube_size = cube_size
        self.offset = offset

        if self.cube_size is None:
            self.cube_size = np.array([0.0515, 0.0515, 0.0515])
        if self.cube_initial_position is None:
            self.cube_initial_position = np.array([0.5, 0.0, 0.0258])
        if self.cube_initial_orientation is None:
            self.cube_initial_orientation = np.array([1, 0, 0, 0])
        if self.target_position is None:
            self.target_position = np.array([-0.3, -0.3, 0.12])
        if self.offset is None:
            self.offset = np.array([0.0, 0.0, 0.0])
        self.target_position = self.target_position + self.offset

        # Create a new USD stage with default sunlight lighting
        stage_utils.create_new_stage(template="sunlight")

        # Create the Franka robot controller (inherits from Articulation)
        self.robot = FrankaExperimental(robot_path="/World/robot", create_robot=True)
        self.end_effector_link = self.robot.end_effector_link

        # Add ground plane environment for physics simulation
        ground_plane = stage_utils.add_reference_to_stage(
            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
            path="/World/ground",
        )

        # Create blue visual material for the cube
        visual_material = PreviewSurfaceMaterial("/Visual_materials/blue")
        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])

        cube_shape = Cube(
            paths="/World/Cube",
            positions=self.cube_initial_position,
            orientations=self.cube_initial_orientation,
            sizes=[1.0],
            scales=self.cube_size,
            reset_xform_op_properties=True,
        )

        cube_geom = GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
        if self.fault_zero_cube_friction:
            zero_friction_material = RigidBodyMaterial(
                "/World/PhysicsMaterials/zero_cube_friction",
                static_frictions=[0.0],
                dynamic_frictions=[0.0],
                restitutions=[0.0],
            )
            zero_friction_material.set_combine_modes(frictions=["min"])
            cube_geom.apply_physics_materials(zero_friction_material)
            self._emit(
                "FAULT_CONFIGURED name=zero-cube-friction "
                "static_friction=0.0 dynamic_friction=0.0 combine=min"
            )
        self.cube = RigidPrim(paths=cube_shape.paths)
        cube_shape.apply_visual_materials(visual_material)

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        """Execute one step of the pick-and-place operation using the specified IK method.

        Args:
            ik_method: The inverse kinematics method to use. Defaults to "damped-least-squares".

        Returns:
            True if a step was executed, False if the sequence is complete.
        """
        if self.is_done():
            return False

        # Hold the reset pose for a few frames so every run starts after the
        # same articulation/cube state has been accepted by the simulator.
        if self._reset_settle_steps_remaining > 0:
            self.robot.reset_to_default_pose()
            self._run_step_count += 1
            self._observe(label="reset_settle")
            self._reset_settle_steps_remaining -= 1
            return True

        self._run_step_count += 1
        self._observe(label="before_action")

        # Get downward-facing orientation for end-effector
        goal_orientation = self.robot.get_downward_orientation()

        # Phase 0: Move to x,y position above cube
        if self._event == 0:
            if self._step == 0:
                self._emit("Phase 0: Moving to x,y position above cube...")
                self._phase_history.append(
                    {
                        "event": self._event,
                        "phase": self.phase_name,
                        "run_step": self._run_step_count,
                    }
                )

            # Use the reset position rather than a live physics pose so that
            # tiny settling differences cannot change the commanded path.
            goal_position = self._run_start_cube_position + np.array([0.0, 0.0, 0.2])

            # Use the new high-level method that combines position and orientation
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)

            self._step += 1
            if self._step >= self.events_dt[0]:
                self._event += 1
                self._step = 0

        # Phase 1: Approach down to the cube
        elif self._event == 1:
            if self._step == 0:
                self._emit("Phase 1: Approaching cube...")
                self._phase_history.append(
                    {
                        "event": self._event,
                        "phase": self.phase_name,
                        "run_step": self._run_step_count,
                    }
                )

            # Approach the fixed cube pose at a repeatable height. The third
            # controlled fault adds only a vertical motion offset here.
            goal_position = self._run_start_cube_position + np.array(
                [0.0, 0.0, 0.1 + self.grasp_height_offset_m]
            )

            # Move to position using the controller
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)

            self._step += 1
            if self._step >= self.events_dt[1]:
                self._event += 1
                self._step = 0

        # Phase 2: Close gripper to grasp the cube
        elif self._event == 2:
            if self._step == 0:
                grasp_command = "open" if self.fault_open_gripper else "close"
                if self.fault_open_gripper:
                    self._emit("FAULT_INJECTED name=open-gripper-during-grasp command=open_gripper")
                elif self.fault_zero_cube_friction:
                    self._emit(
                        "FAULT_INJECTED name=zero-cube-friction "
                        "command=close_gripper static_friction=0.0 dynamic_friction=0.0 combine=min"
                    )
                elif self.fault_high_grasp_approach:
                    self._emit(
                        "FAULT_INJECTED name=high-grasp-approach "
                        "command=close_gripper height_offset_m=0.08 contact=default"
                    )
                self._grasp_ee_distance_at_command = (
                    None
                    if self._last_observation is None
                    else float(self._last_observation["cube_ee_distance"])
                )
                self._emit(f"Phase 2: Grasp command={grasp_command}...")
                self._phase_history.append(
                    {
                        "event": self._event,
                        "phase": self.phase_name,
                        "run_step": self._run_step_count,
                        "grasp_command": grasp_command,
                        "contact_condition": (
                            "zero_cube_friction" if self.fault_zero_cube_friction else "default"
                        ),
                        "approach_condition": (
                            "high_grasp_approach" if self.fault_high_grasp_approach else "default"
                        ),
                        "grasp_height_offset_m": self.grasp_height_offset_m,
                        "grasp_ee_distance_at_command_m": self._grasp_ee_distance_at_command,
                    }
                )

            if self.fault_open_gripper:
                self.robot.open_gripper()
            else:
                self.robot.close_gripper()

            self._step += 1
            if self._step >= self.events_dt[2]:
                self._event += 1
                self._step = 0

        # Phase 3: Lift the cube
        elif self._event == 3:
            if self._step == 0:
                self._emit("Phase 3: Lifting cube...")
                self._phase_history.append(
                    {
                        "event": self._event,
                        "phase": self.phase_name,
                        "run_step": self._run_step_count,
                    }
                )

            # Keep one fixed lift target. The previous implementation added
            # 0.2 m to the *current* pose every frame, which made the target
            # drift upward and made contact outcomes harder to compare.
            goal_position = self._lift_target

            # Move to position using the controller
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)

            self._step += 1
            if self._step >= self.events_dt[3]:
                self._event += 1
                self._step = 0

        # Phase 4: Move cube to target location
        elif self._event == 4:
            if self._step == 0:
                self._emit("Phase 4: Moving cube...")
                self._phase_history.append(
                    {
                        "event": self._event,
                        "phase": self.phase_name,
                        "run_step": self._run_step_count,
                    }
                )

            # Move to position using the controller
            self.robot.set_end_effector_pose(
                position=self.target_position, orientation=goal_orientation, ik_method=ik_method
            )

            self._step += 1
            if self._step >= self.events_dt[4]:
                self._event += 1
                self._step = 0

        # Phase 5: Open gripper to release cube
        elif self._event == 5:
            if self._step == 0:
                self._emit("Phase 5: Opening gripper...")
                self._phase_history.append(
                    {
                        "event": self._event,
                        "phase": self.phase_name,
                        "run_step": self._run_step_count,
                    }
                )

            # Open gripper
            self.robot.open_gripper()

            self._step += 1
            if self._step >= self.events_dt[5]:
                self._event += 1
                self._step = 0

        # Phase 6: Move up
        elif self._event == 6:
            if self._step == 0:
                self._emit("Phase 6: Moving up...")
                self._phase_history.append(
                    {
                        "event": self._event,
                        "phase": self.phase_name,
                        "run_step": self._run_step_count,
                    }
                )

            # Retract from the fixed target rather than from a live cube pose.
            goal_position = self._retract_target

            # Move to position using the controller
            self.robot.set_end_effector_pose(position=goal_position, orientation=goal_orientation, ik_method=ik_method)

            self._step += 1
            if self._step >= self.events_dt[6]:
                self._event += 1
                self._step = 0

        return True

    def is_done(self) -> bool:
        """Check if the pick-and-place sequence is complete.

        Returns:
            True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self.events_dt):
            return True
        else:
            return False

    def reset(self, cube_position: Optional[np.ndarray] = None, cube_orientation: Optional[np.ndarray] = None):
        """Reset the entire pick-and-place system to initial state.

        This is the main reset function that resets both robot and cube.
        Use this for complete system reset.

        Args:
            cube_position: Optional new position for the cube. If None, uses initial position.
            cube_orientation: Optional new orientation for the cube. If None, uses initial orientation.
        """
        self._emit("Resetting pick-and-place system...")
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)
        reset_position = cube_position if cube_position is not None else self.cube_initial_position
        self._run_start_cube_position = np.asarray(reset_position, dtype=float).reshape(3).copy()
        self._lift_target = self._run_start_cube_position + np.array([0.0, 0.0, 0.3])
        self._retract_target = self.target_position + np.array([0.0, 0.0, 0.3])
        self._reset_settle_steps_remaining = self.reset_settle_steps
        self._run_step_count = 0
        self._trace = []
        self._phase_history = []
        self._max_cube_height = float(self._run_start_cube_position[2])
        self._min_cube_ee_distance_during_lift = float("inf")
        self._min_gripper_finger_sum_during_grasp = float("inf")
        self._grasp_ee_distance_at_command = None
        self._last_observation = None
        self._emit("Pick-and-place system reset complete")

    def reset_robot(self):
        """Reset the robot to its default state.

        Resets the robot's joint positions to the default configuration
        and resets the state machine to the beginning.
        """
        if self.robot is not None:
            # Reset robot using the controller
            self.robot.reset_to_default_pose()

            # Reset state machine
            self._event = 0
            self._step = 0

            self._emit("Robot reset to default state")
        else:
            self._emit("Warning: Franka controller not initialized, cannot reset")

    def reset_cube(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None):
        """Reset the cube to its initial position and orientation.

        Args:
            position: Optional new position for the cube. If None, uses initial position.
            orientation: Optional new orientation for the cube. If None, uses initial orientation.
        """
        if self.cube is not None:
            # Use provided position/orientation or fall back to initial values
            reset_position = position if position is not None else self.cube_initial_position
            reset_orientation = orientation if orientation is not None else self.cube_initial_orientation

            # Reset cube position and orientation
            self.cube.set_world_poses(
                positions=reset_position.reshape(1, -1), orientations=reset_orientation.reshape(1, -1)
            )
            self.cube.set_velocities(
                linear_velocities=np.zeros((1, 3)), angular_velocities=np.zeros((1, 3))
            )

            self._emit(f"Cube reset to position: {reset_position}")
        else:
            self._emit("Warning: Cube not initialized, cannot reset")

    def observe(self, label: str = "observe") -> Dict[str, Any]:
        """Record one observation without issuing a robot command."""
        return self._observe(label=label)

    def get_trace(self) -> List[Dict[str, Any]]:
        """Return the per-step trace collected for the current run."""
        return list(self._trace)

    def get_phase_history(self) -> List[Dict[str, Any]]:
        """Return phase-entry records collected for the current run."""
        return list(self._phase_history)

    def evaluate_success(self, xy_tolerance: float = 0.08, z_tolerance: float = 0.06) -> Dict[str, Any]:
        """Evaluate physical placement, not just state-machine completion.

        A successful run must show evidence that the cube was lifted and then
        released near the target on the tabletop. The lift and end-effector
        distance metrics are retained to distinguish contact/grasp failures
        from a placement or success-criterion failure.
        """
        final_observation = self._observe(label="final")
        final_position = np.asarray(final_observation.get("cube_position", [np.nan] * 3), dtype=float)
        target_position = np.asarray(self.target_position, dtype=float)
        tabletop_position = target_position.copy()
        tabletop_position[2] = float(self.cube_size[2]) / 2.0
        final_xy_error = float(np.linalg.norm(final_position[:2] - target_position[:2]))
        final_z_error = float(abs(final_position[2] - tabletop_position[2]))
        max_lift = float(self._max_cube_height - self._run_start_cube_position[2])

        lifted = max_lift >= 0.05
        placed_xy = final_xy_error <= xy_tolerance
        placed_on_table = final_z_error <= z_tolerance
        success = bool(self.is_done() and lifted and placed_xy and placed_on_table)

        reasons = []
        if not self.is_done():
            reasons.append("state_machine_incomplete")
        if not lifted:
            reasons.append("cube_not_lifted")
        if not placed_xy:
            reasons.append("final_xy_out_of_tolerance")
        if not placed_on_table:
            reasons.append("final_z_out_of_tolerance")

        return {
            "success": success,
            "state_machine_complete": bool(self.is_done()),
            "lifted": lifted,
            "placed_xy": placed_xy,
            "placed_on_table": placed_on_table,
            "failure_reasons": reasons,
            "initial_cube_position": self._run_start_cube_position.tolist(),
            "target_position": target_position.tolist(),
            "expected_tabletop_position": tabletop_position.tolist(),
            "final_cube_position": final_position.tolist(),
            "final_xy_error_m": final_xy_error,
            "final_z_error_m": final_z_error,
            "max_cube_height_m": float(self._max_cube_height),
            "max_lift_m": max_lift,
            "min_cube_ee_distance_during_lift_m": (
                None
                if not np.isfinite(self._min_cube_ee_distance_during_lift)
                else float(self._min_cube_ee_distance_during_lift)
            ),
            "min_gripper_finger_sum_during_grasp_m": (
                None
                if not np.isfinite(self._min_gripper_finger_sum_during_grasp)
                else float(self._min_gripper_finger_sum_during_grasp)
            ),
            "grasp_ee_distance_at_command_m": self._grasp_ee_distance_at_command,
            "final_end_effector_position": final_observation.get("end_effector_position"),
        }
