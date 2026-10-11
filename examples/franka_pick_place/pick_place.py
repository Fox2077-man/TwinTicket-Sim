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
import omni.usd
from pxr import UsdPhysics
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
        fault1_variant: str = "normal",
        fault2_variant: str = "normal",
        fault3_variant: str = "normal",
        midway_drop_cause: Optional[str] = None,
        midway_drop_variant: str = "normal",
        misplacement_cause: Optional[str] = None,
        misplacement_variant: str = "normal",
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
            fault1_variant: Explicit Fault 1 comparison variant. ``wrong-cause``
                preserves the open-gripper fault while applying a friction diagnosis;
                ``true-fix`` restores only the close command.
            fault2_variant: Explicit Fault 2 comparison variant.
            fault3_variant: Explicit Fault 3 comparison variant.
            midway_drop_cause: One of ``gripper-release``, ``friction-loss``,
                or ``motion-disturbance`` for the mid-task drop experiment.
            midway_drop_variant: ``injected-fault`` and ``wrong-cause`` keep
                the selected cause active; ``true-fix`` disables it.
            misplacement_cause: One of ``planner-x-offset``,
                ``transport-y-bias``, or ``early-release`` for the inaccurate
                placement experiment.
            misplacement_variant: ``injected-fault`` and ``wrong-cause`` keep
                the selected cause active; ``true-fix`` disables it.
            log_callback: Optional callback for experiment logging.
        """
        self.cube = None
        self.robot = None
        self.reset_settle_steps = max(0, int(reset_settle_steps))
        self.fault_open_gripper = bool(fault_open_gripper)
        self.fault_zero_cube_friction = bool(fault_zero_cube_friction)
        self.fault_high_grasp_approach = bool(fault_high_grasp_approach)
        valid_fault1_variants = {"normal", "injected-fault", "wrong-cause", "true-fix"}
        if fault1_variant not in valid_fault1_variants:
            raise ValueError(f"invalid Fault 1 variant: {fault1_variant}")
        self.fault1_variant = fault1_variant
        self.fault2_variant = fault2_variant
        self.fault3_variant = fault3_variant
        valid_drop_causes = {None, "gripper-release", "friction-loss", "motion-disturbance"}
        valid_variants = {"normal", "injected-fault", "wrong-cause", "true-fix"}
        if midway_drop_cause not in valid_drop_causes or midway_drop_variant not in valid_variants:
            raise ValueError("invalid midway-drop comparison configuration")
        self.midway_drop_cause = midway_drop_cause
        self.midway_drop_variant = midway_drop_variant
        self.midway_drop_active = midway_drop_cause is not None and midway_drop_variant in {
            "injected-fault",
            "wrong-cause",
        }
        valid_misplacement_causes = {None, "planner-x-offset", "transport-y-bias", "early-release"}
        if misplacement_cause not in valid_misplacement_causes or misplacement_variant not in valid_variants:
            raise ValueError("invalid misplacement comparison configuration")
        self.misplacement_cause = misplacement_cause
        self.misplacement_variant = misplacement_variant
        self.misplacement_active = misplacement_cause is not None and misplacement_variant in {
            "injected-fault",
            "wrong-cause",
        }
        if self.midway_drop_active and self.misplacement_active:
            raise ValueError("midway-drop and misplacement faults cannot be active together")
        valid_variants = {"normal", "injected-fault", "wrong-cause", "true-fix"}
        if fault2_variant not in valid_variants or fault3_variant not in valid_variants:
            raise ValueError("invalid controlled fault comparison variant")
        if fault1_variant in {"injected-fault", "wrong-cause"}:
            self.fault_open_gripper = True
        self.fault_zero_cube_friction = self.fault_zero_cube_friction or fault2_variant in {
            "injected-fault",
            "wrong-cause",
        }
        self.fault_high_grasp_approach = self.fault_high_grasp_approach or fault3_variant in {
            "injected-fault",
            "wrong-cause",
        }
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
        self._midway_fault_triggered = False
        self._midway_fault_run_step = None
        self._max_cube_height_before_drop = float("-inf")
        self._min_cube_height_after_drop = float("inf")
        self._cube_height_at_drop = None
        self._cube_ee_distance_at_drop = None
        self._max_cube_ee_distance_after_drop = float("-inf")
        self._drop_action = None
        self._drop_disturbance_velocity_z_mps = None
        self._finger_cube_collision_filter_before = None
        self._finger_cube_collision_filter_after = None
        self._finger_collision_paths = [
            "/World/robot/panda_hand",
            "/World/robot/panda_leftfinger",
            "/World/robot/panda_rightfinger",
        ]
        self._finger_cube_collision_filter_active = False
        self._wrong_diagnosis = None
        self._wrong_diagnosis_action = None
        self._misplacement_fault_triggered = False
        self._misplacement_fault_run_step = None
        self._placement_action = None
        self._planned_release_position = None
        self._commanded_transport_position = None
        self._misplacement_target_offset_m = np.zeros(3)
        self._misplacement_transport_bias_m = np.zeros(3)
        self._early_release_phase_step = 32
        self._early_release_triggered = False
        self._early_release_end_effector_position = None
        self._early_release_cube_position = None
        self._early_release_distance_to_target_m = None
        self._misplacement_wrong_diagnosis = None
        self._misplacement_wrong_diagnosis_action = None
        self.cube_geom = None
        self.finger_collision_geoms = None

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
        if not self._midway_fault_triggered:
            self._max_cube_height_before_drop = max(
                self._max_cube_height_before_drop, float(cube_position[2])
            )
        else:
            self._min_cube_height_after_drop = min(
                self._min_cube_height_after_drop, float(cube_position[2])
            )
            if self._event == 4:
                self._max_cube_ee_distance_after_drop = max(
                    self._max_cube_ee_distance_after_drop, cube_ee_distance
                )
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
        self.cube_geom = cube_geom
        if self.fault_zero_cube_friction:
            zero_friction_material = RigidBodyMaterial(
                "/World/PhysicsMaterials/zero_cube_friction",
                static_frictions=[0.0],
                dynamic_frictions=[0.0],
                restitutions=[0.0],
            )
            zero_friction_material.set_combine_modes(frictions=["min"])
            cube_geom.apply_physics_materials(zero_friction_material)
            if self.fault2_variant == "wrong-cause":
                self._emit(
                    "FAULT_CONFIGURED name=zero-cube-friction "
                    "static_friction=0.0 dynamic_friction=0.0 combine=min "
                    "diagnosis=gripper_command diagnosis_action=repeat_close "
                    "true_cause_preserved=cube_zero_friction"
                )
            else:
                self._emit(
                    "FAULT_CONFIGURED name=zero-cube-friction "
                    "static_friction=0.0 dynamic_friction=0.0 combine=min"
                )
        elif self.fault1_variant == "wrong-cause":
            # Apply a plausible contact diagnosis while preserving the actual
            # open-gripper command. The treatment cannot make an open gripper
            # lift the cube, so a remaining failure is attributable to the
            # unchanged command path rather than the diagnosis.
            diagnosed_friction_material = RigidBodyMaterial(
                "/World/PhysicsMaterials/diagnosed_cube_friction",
                static_frictions=[1.0],
                dynamic_frictions=[1.0],
                restitutions=[0.0],
            )
            diagnosed_friction_material.set_combine_modes(frictions=["max"])
            cube_geom.apply_physics_materials(diagnosed_friction_material)
            self._emit(
                "WRONG_CAUSE_APPLIED suspected=contact_friction action=increase_cube_friction "
                "static_friction=1.0 dynamic_friction=1.0 combine=max "
                "true_cause_preserved=gripper_command_open"
            )
        self.cube = RigidPrim(paths=cube_shape.paths)
        cube_shape.apply_visual_materials(visual_material)

    def _trigger_midway_drop(self) -> None:
        """Apply one mid-transport intervention after the cube was grasped."""
        if not self.midway_drop_active or self._midway_fault_triggered:
            return

        self._midway_fault_triggered = True
        self._midway_fault_run_step = self._run_step_count
        self._cube_height_at_drop = (
            None
            if self._last_observation is None
            else float(self._last_observation.get("cube_position", [0.0, 0.0, 0.0])[2])
        )
        self._cube_ee_distance_at_drop = (
            None
            if self._last_observation is None
            else float(self._last_observation.get("cube_ee_distance"))
        )
        cause = self.midway_drop_cause
        wrong = self.midway_drop_variant == "wrong-cause"
        if wrong:
            self._wrong_diagnosis = {
                "gripper-release": "contact_friction",
                "friction-loss": "gripper_command",
                "motion-disturbance": "gripper_command",
            }[cause]
            self._wrong_diagnosis_action = {
                "gripper-release": "increase_cube_friction",
                "friction-loss": "repeat_close",
                "motion-disturbance": "verify_close_command",
            }[cause]
            self._emit(
                f"WRONG_CAUSE_APPLIED cause={cause} diagnosis={self._wrong_diagnosis} "
                f"action={self._wrong_diagnosis_action} "
                "true_cause_preserved=true"
            )
            # These treatments are intentionally ineffective. The real
            # intervention below remains active in the same simulation step.
            if self._wrong_diagnosis_action == "repeat_close":
                self.robot.close_gripper()
            elif self._wrong_diagnosis_action == "increase_cube_friction":
                diagnosed_friction_material = RigidBodyMaterial(
                    "/World/PhysicsMaterials/midway_diagnosed_friction",
                    static_frictions=[1.0],
                    dynamic_frictions=[1.0],
                    restitutions=[0.0],
                )
                diagnosed_friction_material.set_combine_modes(frictions=["max"])
                self.cube_geom.apply_physics_materials(diagnosed_friction_material)
            else:
                # Verifying the already-issued close command is diagnostic only.
                # It must not alter contact physics or the active disturbance.
                assert self._wrong_diagnosis_action == "verify_close_command"

        if cause == "gripper-release":
            self._drop_action = "open_gripper_during_transport"
            self._emit(
                "FAULT_INJECTED name=midway-gripper-release phase=move_to_target "
                "action=open_gripper true_cause_preserved=true" if wrong else
                "FAULT_INJECTED name=midway-gripper-release phase=move_to_target action=open_gripper"
            )
            self.robot.open_gripper()
        elif cause == "friction-loss":
            self._drop_action = "disable_finger_cube_collision_during_transport"
            # Filter only cube-versus-gripper collision pairs. Unlike changing
            # the cube material or state, this leaves gravity, the table, and
            # robot commands untouched while removing the physical constraint
            # that carries the cube.
            stage = omni.usd.get_context().get_stage()
            cube_prim = stage.GetPrimAtPath("/World/Cube")
            filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(cube_prim)
            relation = filtered_pairs.CreateFilteredPairsRel()
            for path in self._finger_collision_paths:
                relation.AddTarget(path)
            self._finger_cube_collision_filter_before = [False] * len(self._finger_collision_paths)
            self._finger_cube_collision_filter_after = [True] * len(self._finger_collision_paths)
            self._finger_cube_collision_filter_active = True
            before_text = "[false,false,false]"
            after_text = "[true,true,true]"
            self._emit(
                "FAULT_INJECTED name=midway-finger-cube-collision-disabled "
                "phase=move_to_target finger_collision_paths=panda_hand,panda_leftfinger,panda_rightfinger "
                "mechanism=UsdPhysics.FilteredPairsAPI "
                f"collision_filter_before={before_text} "
                f"collision_filter_after={after_text} "
                "true_cause_preserved=true"
                if wrong
                else "FAULT_INJECTED name=midway-finger-cube-collision-disabled "
                "phase=move_to_target finger_collision_paths=panda_hand,panda_leftfinger,panda_rightfinger "
                "mechanism=UsdPhysics.FilteredPairsAPI "
                f"collision_filter_before={before_text} "
                f"collision_filter_after={after_text}"
            )
        elif cause == "motion-disturbance":
            self._drop_action = "downward_disturbance_velocity"
            self._drop_disturbance_velocity_z_mps = -3.0
            self.cube.set_velocities(
                linear_velocities=np.array([[0.0, 0.0, self._drop_disturbance_velocity_z_mps]]),
                angular_velocities=np.zeros((1, 3)),
            )
            self._emit(
                "FAULT_INJECTED name=midway-motion-disturbance phase=move_to_target "
                "linear_velocity_z_mps=-3.0 true_cause_preserved=true"
                if wrong
                else "FAULT_INJECTED name=midway-motion-disturbance phase=move_to_target linear_velocity_z_mps=-3.0"
            )

    def _configure_misplacement_transport(self) -> None:
        """Set one placement-only intervention at the transport phase boundary."""
        target = np.asarray(self.target_position, dtype=float)
        self._planned_release_position = target.copy()
        self._commanded_transport_position = target.copy()
        self._misplacement_target_offset_m = np.zeros(3)
        self._misplacement_transport_bias_m = np.zeros(3)

        if not self.misplacement_active or self._misplacement_fault_triggered:
            return

        self._misplacement_fault_triggered = True
        self._misplacement_fault_run_step = self._run_step_count
        cause = self.misplacement_cause
        wrong = self.misplacement_variant == "wrong-cause"
        if wrong:
            self._misplacement_wrong_diagnosis = {
                "planner-x-offset": "gripper_closure",
                "transport-y-bias": "planner_target_frame",
                "early-release": "transport_tracking",
            }[cause]
            self._misplacement_wrong_diagnosis_action = {
                "planner-x-offset": "repeat_close_gripper",
                "transport-y-bias": "reissue_nominal_planner_target",
                "early-release": "reissue_nominal_transport_target",
            }[cause]
            self._emit(
                f"WRONG_CAUSE_APPLIED cause={cause} diagnosis={self._misplacement_wrong_diagnosis} "
                f"action={self._misplacement_wrong_diagnosis_action} true_cause_preserved=true"
            )
            if self._misplacement_wrong_diagnosis_action == "repeat_close_gripper":
                self.robot.close_gripper()

        if cause == "planner-x-offset":
            self._placement_action = "planner_release_target_x_offset"
            self._misplacement_target_offset_m = np.array([0.16, 0.0, 0.0])
            self._planned_release_position = target + self._misplacement_target_offset_m
            self._commanded_transport_position = self._planned_release_position.copy()
        elif cause == "transport-y-bias":
            self._placement_action = "transport_controller_y_command_bias"
            self._misplacement_transport_bias_m = np.array([0.0, 0.16, 0.0])
            self._commanded_transport_position = target + self._misplacement_transport_bias_m
        elif cause == "early-release":
            self._placement_action = "open_gripper_before_target"
        else:
            raise RuntimeError(f"unsupported misplacement cause: {cause}")

        self._emit(
            f"FAULT_INJECTED name=misplacement-{cause} phase=move_to_target "
            f"action={self._placement_action} planned_release_position_m={self._planned_release_position.tolist()} "
            f"commanded_transport_position_m={self._commanded_transport_position.tolist()} "
            f"planner_offset_m={self._misplacement_target_offset_m.tolist()} "
            f"transport_bias_m={self._misplacement_transport_bias_m.tolist()} "
            f"early_release_phase_step={self._early_release_phase_step}"
            + (" true_cause_preserved=true" if wrong else "")
        )

    def _trigger_early_release(self) -> None:
        """Open the gripper once at a measured, pre-target transport step."""
        if self._early_release_triggered or self.misplacement_cause != "early-release":
            return
        self._early_release_triggered = True
        observation = self._last_observation or {}
        end_effector = observation.get("end_effector_position")
        cube_position = observation.get("cube_position")
        self._early_release_end_effector_position = None if end_effector is None else list(end_effector)
        self._early_release_cube_position = None if cube_position is None else list(cube_position)
        if end_effector is not None:
            self._early_release_distance_to_target_m = float(
                np.linalg.norm(np.asarray(end_effector, dtype=float)[:2] - self.target_position[:2])
            )
        self._emit(
            "FAULT_INJECTED name=misplacement-early-release phase=move_to_target "
            f"phase_step={self._step} end_effector_position_m={self._early_release_end_effector_position} "
            f"cube_position_m={self._early_release_cube_position} "
            f"distance_to_target_xy_m={self._early_release_distance_to_target_m}"
        )
        self._phase_history.append(
            {
                "event": self._event,
                "phase": "early_release_fault",
                "run_step": self._run_step_count,
                "phase_step": self._step,
                "end_effector_position_m": self._early_release_end_effector_position,
                "cube_position_m": self._early_release_cube_position,
                "distance_to_target_xy_m": self._early_release_distance_to_target_m,
            }
        )
        self.robot.open_gripper()

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
                if self.fault1_variant == "wrong-cause":
                    self._emit(
                        "FAULT_INJECTED name=open-gripper-during-grasp command=open_gripper "
                        "diagnosis=contact_friction diagnosis_action=increase_cube_friction"
                    )
                elif self.fault2_variant == "wrong-cause":
                    self._emit(
                        "FAULT_INJECTED name=zero-cube-friction command=close_gripper "
                        "diagnosis=gripper_command diagnosis_action=repeat_close "
                        "true_cause_preserved=cube_zero_friction"
                    )
                elif self.fault_open_gripper:
                    self._emit("FAULT_INJECTED name=open-gripper-during-grasp command=open_gripper")
                elif self.fault_zero_cube_friction:
                    self._emit(
                        "FAULT_INJECTED name=zero-cube-friction "
                        "command=close_gripper static_friction=0.0 dynamic_friction=0.0 combine=min"
                    )
                elif self.fault_high_grasp_approach:
                    if self.fault3_variant == "wrong-cause":
                        self._emit(
                            "FAULT_INJECTED name=high-grasp-approach command=close_gripper "
                            "diagnosis=height_reference_calibration "
                            "diagnosis_action=verify_cube_top_height "
                            "height_offset_m=0.08 true_cause_preserved=approach_geometry"
                        )
                    else:
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
                            "zero_cube_friction"
                            if self.fault_zero_cube_friction
                            else (
                                "increased_cube_friction"
                                if self.fault1_variant == "wrong-cause"
                                else "default"
                            )
                        ),
                        "approach_condition": (
                            "high_grasp_approach" if self.fault_high_grasp_approach else "default"
                        ),
                        "fault2_contact_condition": (
                            "zero_cube_friction"
                            if self.fault_zero_cube_friction
                            else "default"
                        ),
                        "fault3_approach_condition": (
                            "high_grasp_approach"
                            if self.fault_high_grasp_approach
                            else "default"
                        ),
                        "wrong_diagnosis": (
                            "height_reference_calibration"
                            if self.fault3_variant == "wrong-cause"
                            else None
                        ),
                        "wrong_diagnosis_action": (
                            "verify_cube_top_height"
                            if self.fault3_variant == "wrong-cause"
                            else None
                        ),
                        "grasp_height_offset_m": self.grasp_height_offset_m,
                        "grasp_ee_distance_at_command_m": self._grasp_ee_distance_at_command,
                        "fault1_variant": self.fault1_variant,
                        "fault2_variant": self.fault2_variant,
                        "fault3_variant": self.fault3_variant,
                        "true_cause_preserved": any(
                            variant == "wrong-cause"
                            for variant in (self.fault1_variant, self.fault2_variant, self.fault3_variant)
                        ),
                        "true_cause_modified": any(
                            variant == "true-fix"
                            for variant in (self.fault1_variant, self.fault2_variant, self.fault3_variant)
                        ),
                    }
                )

            if self.fault_open_gripper:
                self.robot.open_gripper()
            else:
                self.robot.close_gripper()
                if self.fault2_variant == "wrong-cause":
                    # The diagnosed command path is deliberately ineffective;
                    # repeat the normal close command while retaining the real
                    # zero-friction cause.
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
                if self.midway_drop_active:
                    self._trigger_midway_drop()
                self._configure_misplacement_transport()
                self._phase_history.append(
                    {
                        "event": self._event,
                        "phase": self.phase_name,
                        "run_step": self._run_step_count,
                        "midway_drop_cause": self.midway_drop_cause,
                        "midway_drop_variant": self.midway_drop_variant,
                        "drop_triggered": self._midway_fault_triggered,
                        "drop_action": self._drop_action,
                        "drop_disturbance_velocity_z_mps": self._drop_disturbance_velocity_z_mps,
                        "true_cause_preserved": self.midway_drop_variant == "wrong-cause",
                        "true_cause_modified": self.midway_drop_variant == "true-fix",
                        "wrong_diagnosis": self._wrong_diagnosis,
                        "wrong_diagnosis_action": self._wrong_diagnosis_action,
                        "cube_height_at_drop_m": self._cube_height_at_drop,
                        "finger_cube_collision_filter_before": self._finger_cube_collision_filter_before,
                        "finger_cube_collision_filter_after": self._finger_cube_collision_filter_after,
                        "finger_cube_collision_filter_targets": self._finger_collision_paths,
                        "max_cube_height_before_drop_m": (
                            None
                            if not np.isfinite(self._max_cube_height_before_drop)
                            else float(self._max_cube_height_before_drop)
                        ),
                        "misplacement_cause": self.misplacement_cause,
                        "misplacement_variant": self.misplacement_variant,
                        "misplacement_triggered": self._misplacement_fault_triggered,
                        "placement_action": self._placement_action,
                        "planned_release_position_m": (
                            None if self._planned_release_position is None else self._planned_release_position.tolist()
                        ),
                        "commanded_transport_position_m": (
                            None
                            if self._commanded_transport_position is None
                            else self._commanded_transport_position.tolist()
                        ),
                        "planner_offset_m": self._misplacement_target_offset_m.tolist(),
                        "transport_command_bias_m": self._misplacement_transport_bias_m.tolist(),
                        "early_release_phase_step": self._early_release_phase_step,
                        "misplacement_wrong_diagnosis": self._misplacement_wrong_diagnosis,
                        "misplacement_wrong_diagnosis_action": self._misplacement_wrong_diagnosis_action,
                        "misplacement_true_cause_preserved": self.misplacement_variant == "wrong-cause",
                        "misplacement_true_cause_modified": self.misplacement_variant == "true-fix",
                    }
                )

            if self._midway_fault_triggered and self.midway_drop_cause == "gripper-release":
                self.robot.open_gripper()
            if (
                self._misplacement_fault_triggered
                and self.misplacement_cause == "early-release"
                and self._step >= self._early_release_phase_step
            ):
                self._trigger_early_release()

            # Move to position using the controller
            self.robot.set_end_effector_pose(
                position=self._commanded_transport_position, orientation=goal_orientation, ik_method=ik_method
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
        self._midway_fault_triggered = False
        self._midway_fault_run_step = None
        self._max_cube_height_before_drop = float("-inf")
        self._min_cube_height_after_drop = float("inf")
        self._cube_height_at_drop = None
        self._cube_ee_distance_at_drop = None
        self._max_cube_ee_distance_after_drop = float("-inf")
        self._drop_action = None
        self._drop_disturbance_velocity_z_mps = None
        self._finger_cube_collision_filter_before = None
        self._finger_cube_collision_filter_after = None
        self._finger_cube_collision_filter_active = False
        self._wrong_diagnosis = None
        self._wrong_diagnosis_action = None
        self._misplacement_fault_triggered = False
        self._misplacement_fault_run_step = None
        self._placement_action = None
        self._planned_release_position = None
        self._commanded_transport_position = None
        self._misplacement_target_offset_m = np.zeros(3)
        self._misplacement_transport_bias_m = np.zeros(3)
        self._early_release_triggered = False
        self._early_release_end_effector_position = None
        self._early_release_cube_position = None
        self._early_release_distance_to_target_m = None
        self._misplacement_wrong_diagnosis = None
        self._misplacement_wrong_diagnosis_action = None
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
        pre_drop_height = (
            self._max_cube_height_before_drop
            if np.isfinite(self._max_cube_height_before_drop)
            else self._max_cube_height
        )
        pre_drop_lift = float(pre_drop_height - self._run_start_cube_position[2])
        post_drop_min_height = (
            None
            if not np.isfinite(self._min_cube_height_after_drop)
            else float(self._min_cube_height_after_drop)
        )
        drop_distance = (
            None
            if post_drop_min_height is None
            else float(pre_drop_height - post_drop_min_height)
        )
        post_drop_max_separation = (
            None
            if not np.isfinite(self._max_cube_ee_distance_after_drop)
            else float(self._max_cube_ee_distance_after_drop)
        )
        separation_increase = (
            None
            if self._cube_ee_distance_at_drop is None or post_drop_max_separation is None
            else float(post_drop_max_separation - self._cube_ee_distance_at_drop)
        )
        grasp_established_before_fault = pre_drop_lift >= 0.05
        drop_detected = bool(
            self._midway_fault_triggered
            and grasp_established_before_fault
            and drop_distance is not None
            and drop_distance >= 0.05
            and separation_increase is not None
            and separation_increase >= 0.08
        )

        lifted = max_lift >= 0.05
        placed_xy = final_xy_error <= xy_tolerance
        placed_on_table = final_z_error <= z_tolerance
        misplacement_detected = bool(
            self._misplacement_fault_triggered and lifted and final_xy_error > xy_tolerance
        )
        # A mid-transport drop is a task failure even when the cube later
        # happens to settle inside the target tolerances.  Treat the physical
        # event as authoritative so the comparison cannot call a dropped run
        # successful merely because its final pose looks acceptable.
        success = bool(
            self.is_done()
            and lifted
            and placed_xy
            and placed_on_table
            and not drop_detected
            and not misplacement_detected
        )

        reasons = []
        if not self.is_done():
            reasons.append("state_machine_incomplete")
        if not lifted:
            reasons.append("cube_not_lifted")
        if not placed_xy:
            reasons.append("final_xy_out_of_tolerance")
        if not placed_on_table:
            reasons.append("final_z_out_of_tolerance")
        if drop_detected:
            reasons.append("midway_drop_detected")
        if misplacement_detected:
            reasons.append("misplacement_detected")

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
            "final_xy_error_vector_m": (final_position[:2] - target_position[:2]).tolist(),
            "final_xy_error_m": final_xy_error,
            "final_z_error_m": final_z_error,
            "max_cube_height_m": float(self._max_cube_height),
            "max_lift_m": max_lift,
            "midway_drop_cause": self.midway_drop_cause,
            "midway_drop_variant": self.midway_drop_variant,
            "midway_fault_triggered": self._midway_fault_triggered,
            "midway_fault_run_step": self._midway_fault_run_step,
            "drop_action": self._drop_action,
            "drop_disturbance_velocity_z_mps": self._drop_disturbance_velocity_z_mps,
            "finger_cube_collision_filter_before": self._finger_cube_collision_filter_before,
            "finger_cube_collision_filter_after": self._finger_cube_collision_filter_after,
            "finger_cube_collision_filter_targets": self._finger_collision_paths,
            "cube_height_at_drop_m": self._cube_height_at_drop,
            "cube_ee_distance_at_drop_m": self._cube_ee_distance_at_drop,
            "max_cube_height_before_drop_m": float(pre_drop_height),
            "pre_drop_lift_m": pre_drop_lift,
            "min_cube_height_after_drop_m": post_drop_min_height,
            "drop_distance_m": drop_distance,
            "max_cube_ee_distance_after_drop_m": post_drop_max_separation,
            "drop_separation_increase_m": separation_increase,
            "grasp_established_before_fault": grasp_established_before_fault,
            "drop_detected": drop_detected,
            "midway_wrong_diagnosis": self._wrong_diagnosis,
            "midway_wrong_diagnosis_action": self._wrong_diagnosis_action,
            "misplacement_cause": self.misplacement_cause,
            "misplacement_variant": self.misplacement_variant,
            "misplacement_fault_triggered": self._misplacement_fault_triggered,
            "misplacement_fault_run_step": self._misplacement_fault_run_step,
            "misplacement_detected": misplacement_detected,
            "placement_action": self._placement_action,
            "planned_release_position_m": (
                None if self._planned_release_position is None else self._planned_release_position.tolist()
            ),
            "commanded_transport_position_m": (
                None if self._commanded_transport_position is None else self._commanded_transport_position.tolist()
            ),
            "planner_offset_m": self._misplacement_target_offset_m.tolist(),
            "transport_command_bias_m": self._misplacement_transport_bias_m.tolist(),
            "early_release_phase_step": self._early_release_phase_step,
            "early_release_triggered": self._early_release_triggered,
            "early_release_end_effector_position_m": self._early_release_end_effector_position,
            "early_release_cube_position_m": self._early_release_cube_position,
            "early_release_distance_to_target_xy_m": self._early_release_distance_to_target_m,
            "misplacement_wrong_diagnosis": self._misplacement_wrong_diagnosis,
            "misplacement_wrong_diagnosis_action": self._misplacement_wrong_diagnosis_action,
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
