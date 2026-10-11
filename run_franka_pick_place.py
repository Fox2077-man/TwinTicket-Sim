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

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without a visible Isaac Sim window")
parser.add_argument("--webrtc", action="store_true", help="Enable Isaac Sim WebRTC livestreaming")
parser.add_argument("--test", action="store_true", help="Exit after one completed pick-and-place sequence")
parser.add_argument(
    "--auto-start",
    action="store_true",
    help="Start immediately instead of waiting for the Start/Pause UI button",
)
parser.add_argument(
    "--repeat",
    type=int,
    default=1,
    metavar="N",
    help="Number of pick-and-place runs; use 0 for infinite repetition (default: 1)",
)
parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Simulation device")
parser.add_argument(
    "--ik-method",
    type=str,
    choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
    default="damped-least-squares",
    help="Differential inverse kinematics method",
)
parser.add_argument("--seed", type=int, default=0, help="Seed Python and NumPy random generators")
parser.add_argument(
    "--start-condition-id",
    default="default",
    help="Stable identifier recorded with the initial scene condition",
)
parser.add_argument(
    "--cube-initial-position",
    type=float,
    nargs=3,
    metavar=("X", "Y", "Z"),
    default=None,
    help="Override the cube initial position in meters",
)
parser.add_argument(
    "--reset-settle-steps",
    type=int,
    default=15,
    help="Frames to hold the reset state before each run",
)
parser.add_argument(
    "--post-run-settle-steps",
    type=int,
    default=30,
    help="Frames to let the released cube settle before evaluating success",
)
parser.add_argument(
    "--results-dir",
    type=Path,
    default=Path("artifacts/baseline"),
    help="Directory under which a timestamped experiment session is stored",
)
parser.add_argument(
    "--record-video",
    action="store_true",
    help="Record one viewport MP4 per run (requires an active viewport, normally --webrtc)",
)
fault_group = parser.add_mutually_exclusive_group()
fault_group.add_argument(
    "--fault-open-gripper",
    action="store_true",
    help="Inject the first controlled fault: keep the gripper open during the grasp phase",
)
fault_group.add_argument(
    "--fault-zero-cube-friction",
    action="store_true",
    help="Inject the second controlled fault: remove friction from the cube contact surface",
)
fault_group.add_argument(
    "--fault-high-grasp-approach",
    action="store_true",
    help="Inject the third controlled fault: approach the grasp pose 0.08 m too high",
)
fault_group.add_argument(
    "--fault1-variant",
    choices=["normal", "injected-fault", "wrong-cause", "true-fix"],
    default=None,
    help=(
        "Run one reproducible Fault 1 comparison variant. The wrong-cause variant "
        "preserves the open-gripper fault while applying a plausible friction diagnosis; "
        "true-fix restores only the close command."
    ),
)
fault_group.add_argument(
    "--fault1-wrong-cause",
    dest="fault1_wrong_cause",
    action="store_true",
    help="Alias for --fault1-variant wrong-cause",
)
fault_group.add_argument(
    "--fault1-true-fix",
    dest="fault1_true_fix",
    action="store_true",
    help="Alias for --fault1-variant true-fix",
)
fault_group.add_argument(
    "--fault2-variant",
    choices=["normal", "injected-fault", "wrong-cause", "true-fix"],
    default=None,
    help=(
        "Run one reproducible Fault 2 comparison variant. The wrong-cause variant "
        "repeats the close command while preserving zero cube friction; true-fix "
        "restores the default cube contact material."
    ),
)
fault_group.add_argument(
    "--fault3-variant",
    choices=["normal", "injected-fault", "wrong-cause", "true-fix"],
    default=None,
    help=(
        "Run one reproducible Fault 3 comparison variant. The wrong-cause variant "
        "checks the height reference while preserving the high approach; true-fix "
        "restores the normal grasp height."
    ),
)
parser.add_argument(
    "--midway-drop-cause",
    choices=["gripper-release", "friction-loss", "motion-disturbance"],
    default=None,
    help="Cause used by the mid-transport drop comparison",
)
parser.add_argument(
    "--midway-drop-variant",
    choices=["normal", "injected-fault", "wrong-cause", "true-fix"],
    default="normal",
    help="Variant used by the mid-transport drop comparison",
)
parser.add_argument(
    "--misplacement-cause",
    choices=["planner-x-offset", "transport-y-bias", "early-release"],
    default=None,
    help="Cause used by the inaccurate-placement comparison",
)
parser.add_argument(
    "--misplacement-variant",
    choices=["normal", "injected-fault", "wrong-cause", "true-fix"],
    default="normal",
    help="Variant used by the inaccurate-placement comparison",
)
args, _ = parser.parse_known_args()
if args.repeat < 0:
    parser.error("--repeat must be 0 or a positive integer")
if args.reset_settle_steps < 0 or args.post_run_settle_steps < 0:
    parser.error("settle step counts must be non-negative")

random.seed(args.seed)
np.random.seed(args.seed)


def resolve_fault1_variant(parsed_args: argparse.Namespace) -> str:
    """Normalize explicit Fault 1 comparison aliases without changing legacy flags."""
    if parsed_args.fault1_variant is not None:
        return parsed_args.fault1_variant
    if parsed_args.fault1_wrong_cause:
        return "wrong-cause"
    if parsed_args.fault1_true_fix:
        return "true-fix"
    if parsed_args.fault_open_gripper:
        return "injected-fault"
    return "normal"


fault1_variant = resolve_fault1_variant(args)


def resolve_fault_variant(
    explicit_variant: Optional[str], legacy_fault: bool, fault_number: int
) -> str:
    if explicit_variant is not None:
        return explicit_variant
    if legacy_fault:
        return "injected-fault"
    return "normal"


fault2_variant = resolve_fault_variant(args.fault2_variant, args.fault_zero_cube_friction, 2)
fault3_variant = resolve_fault_variant(args.fault3_variant, args.fault_high_grasp_approach, 3)
comparison_variants = [
    ("fault1", fault1_variant),
    ("fault2", fault2_variant),
    ("fault3", fault3_variant),
]
active_comparisons = [(fault, variant) for fault, variant in comparison_variants if variant != "normal"]
if args.midway_drop_variant != "normal":
    active_comparisons.append(("midway-drop", args.midway_drop_variant))
if args.misplacement_variant != "normal":
    active_comparisons.append(("misplacement", args.misplacement_variant))
if len(active_comparisons) > 1:
    parser.error("only one fault comparison variant may be active per run")
if args.midway_drop_variant != "normal" and args.midway_drop_cause is None:
    parser.error("--midway-drop-cause is required when --midway-drop-variant is not normal")
if args.misplacement_variant != "normal" and args.misplacement_cause is None:
    parser.error("--misplacement-cause is required when --misplacement-variant is not normal")


class ExperimentLogger:
    """Line-buffered session logger that survives Kit's non-TTY stdout behavior."""

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.session_dir / "experiment.log"
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"{timestamp} {message}"
        self._stream.write(line + "\n")
        self._stream.flush()
        print(line, flush=True)

    def close(self) -> None:
        self._stream.close()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_trace(path: Path, trace: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in trace:
            stream.write(json.dumps(item, sort_keys=True) + "\n")


def copy_latest_kit_log(session_dir: Path) -> Optional[Path]:
    """Copy the Isaac Sim Kit log into the mounted experiment directory."""
    try:
        candidates = []
        for root in (Path("/isaac-sim/kit/logs"), Path("/isaac-sim/.nvidia-omniverse/logs")):
            if root.exists():
                candidates.extend(root.rglob("kit_*.log"))
        if not candidates:
            return None
        source = max(candidates, key=lambda item: item.stat().st_mtime)
        destination = session_dir / "kit.log"
        shutil.copy2(source, destination)
        return destination
    except Exception:
        return None


from isaacsim import SimulationApp

simulation_config = {"headless": args.headless or args.webrtc}
if args.webrtc:
    # WebRTC needs a headless renderer while keeping the streamed UI visible.
    simulation_config["hide_ui"] = False

session_name = datetime.now().strftime("session-%Y%m%d-%H%M%S")
session_dir = args.results_dir / session_name
logger = ExperimentLogger(session_dir)
if args.fault_open_gripper:
    fault_name = "open-gripper-during-grasp"
    fault_details = {"cause_path": "gripper_command", "grasp_command": "open"}
elif args.fault_zero_cube_friction:
    fault_name = "zero-cube-friction"
    fault_details = {
        "cause_path": "contact_friction",
        "cube_static_friction": 0.0,
        "cube_dynamic_friction": 0.0,
        "friction_combine_mode": "min",
    }
elif args.fault_high_grasp_approach:
    fault_name = "high-grasp-approach"
    fault_details = {
        "cause_path": "approach_geometry",
        "grasp_height_offset_m": 0.08,
        "grasp_command": "close",
        "contact_condition": "default",
    }
else:
    fault_name = None
    fault_details = {}
if fault1_variant == "injected-fault":
    fault_name = "open-gripper-during-grasp"
    fault_details = {
        "cause_path": "gripper_command",
        "grasp_command": "open",
        "true_cause": "gripper_command_open",
    }
elif fault1_variant == "wrong-cause":
    # The wrong diagnosis is deliberately applied alongside the unchanged open
    # command. This makes it impossible for error handling to mask the true cause.
    fault_name = "open-gripper-during-grasp"
    fault_details = {
        "cause_path": "gripper_command",
        "grasp_command": "open",
        "true_cause": "gripper_command_open",
        "diagnosis": "contact_friction",
        "diagnosis_action": "increase_cube_friction",
        "cube_static_friction": 1.0,
        "cube_dynamic_friction": 1.0,
        "friction_combine_mode": "max",
        "true_cause_preserved": True,
    }
elif fault1_variant == "true-fix":
    fault_name = None
    fault_details = {
        "cause_path": "gripper_command",
        "grasp_command": "close",
        "repair": "restore_close_gripper",
        "true_cause_modified": True,
    }
if fault2_variant == "injected-fault":
    fault_name = "zero-cube-friction"
    fault_details = {
        "cause_path": "contact_friction",
        "cube_static_friction": 0.0,
        "cube_dynamic_friction": 0.0,
        "friction_combine_mode": "min",
        "true_cause": "cube_zero_friction",
    }
elif fault2_variant == "wrong-cause":
    fault_name = "zero-cube-friction"
    fault_details = {
        "cause_path": "contact_friction",
        "cube_static_friction": 0.0,
        "cube_dynamic_friction": 0.0,
        "friction_combine_mode": "min",
        "true_cause": "cube_zero_friction",
        "diagnosis": "height_reference_calibration",
        "diagnosis_action": "verify_cube_top_height",
        "true_cause_preserved": True,
    }
elif fault2_variant == "true-fix":
    fault_name = None
    fault_details = {
        "cause_path": "contact_friction",
        "repair": "restore_default_cube_contact_material",
        "true_cause_modified": True,
    }
if fault3_variant == "injected-fault":
    fault_name = "high-grasp-approach"
    fault_details = {
        "cause_path": "approach_geometry",
        "grasp_height_offset_m": 0.08,
        "grasp_command": "close",
        "contact_condition": "default",
        "true_cause": "approach_height_offset",
    }
elif fault3_variant == "wrong-cause":
    fault_name = "high-grasp-approach"
    fault_details = {
        "cause_path": "approach_geometry",
        "grasp_height_offset_m": 0.08,
        "grasp_command": "close",
        "contact_condition": "default",
        "true_cause": "approach_height_offset",
        "diagnosis": "height_reference_calibration",
        "diagnosis_action": "verify_cube_top_height",
        "true_cause_preserved": True,
    }
elif fault3_variant == "true-fix":
    fault_name = None
    fault_details = {
        "cause_path": "approach_geometry",
        "repair": "restore_normal_grasp_height",
        "true_cause_modified": True,
    }
if args.midway_drop_cause is not None:
    midway_cause_details = {
        "gripper-release": {
            "true_cause": "gripper_command_open_during_transport",
            "wrong_diagnosis": "contact_friction",
            "wrong_diagnosis_action": "increase_cube_friction",
        },
        "friction-loss": {
            "true_cause": "finger_cube_collision_disabled_during_transport",
            "wrong_diagnosis": "gripper_command",
            "wrong_diagnosis_action": "repeat_close",
        },
        "motion-disturbance": {
            "true_cause": "downward_disturbance_velocity",
            "wrong_diagnosis": "gripper_command",
            "wrong_diagnosis_action": "verify_close_command",
        },
    }[args.midway_drop_cause]
    fault_name = f"midway-drop-{args.midway_drop_cause}"
    fault_details = {
        "cause_path": f"midway_drop.{args.midway_drop_cause}",
        "midway_drop_cause": args.midway_drop_cause,
        "midway_drop_variant": args.midway_drop_variant,
        "true_cause": midway_cause_details["true_cause"],
        "true_cause_preserved": args.midway_drop_variant == "wrong-cause",
        "true_cause_modified": args.midway_drop_variant == "true-fix",
    }
    if args.midway_drop_variant == "wrong-cause":
        fault_details.update({
            "diagnosis": midway_cause_details["wrong_diagnosis"],
            "diagnosis_action": midway_cause_details["wrong_diagnosis_action"],
        })
    elif args.midway_drop_variant == "true-fix":
        fault_details["repair"] = f"remove_{midway_cause_details['true_cause']}"
if args.misplacement_cause is not None:
    cause_details = {
        "planner-x-offset": {
            "true_cause": "planner_release_target_x_offset",
            "planner_offset_m": [0.16, 0.0, 0.0],
        },
        "transport-y-bias": {
            "true_cause": "transport_controller_y_command_bias",
            "transport_command_bias_m": [0.0, 0.16, 0.0],
        },
        "early-release": {
            "true_cause": "open_gripper_before_target",
            "early_release_phase_step": 32,
        },
    }[args.misplacement_cause]
    fault_name = f"misplacement-{args.misplacement_cause}"
    fault_details = {
        "cause_path": f"misplacement.{args.misplacement_cause}",
        "misplacement_cause": args.misplacement_cause,
        "misplacement_variant": args.misplacement_variant,
        **cause_details,
        "true_cause_preserved": args.misplacement_variant == "wrong-cause",
        "true_cause_modified": args.misplacement_variant == "true-fix",
    }
logger.log(
    f"SESSION_START dir={session_dir} seed={args.seed} repeat={args.repeat} "
    f"start_condition_id={args.start_condition_id} "
    f"cube_initial_position={args.cube_initial_position or 'default'} "
    f"device={args.device} fault={fault_name or 'none'} "
    f"fault1_variant={fault1_variant} fault2_variant={fault2_variant} fault3_variant={fault3_variant} "
    f"midway_drop_cause={args.midway_drop_cause or 'none'} midway_drop_variant={args.midway_drop_variant} "
    f"misplacement_cause={args.misplacement_cause or 'none'} misplacement_variant={args.misplacement_variant}"
)

simulation_app = SimulationApp(simulation_config)

if args.webrtc or args.record_video:
    from isaacsim.core.utils.extensions import enable_extension

    if args.webrtc:
        enable_extension("omni.services.livestream.nvcf")
    if args.record_video:
        enable_extension("omni.kit.capture.viewport")
        enable_extension("omni.videoencoding")
    simulation_app.update()

import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from examples.franka_pick_place.pick_place import FrankaPickPlace

execution_control = None


class ExecutionControl:
    """Small Isaac Sim UI window for starting and pausing the state machine."""

    def __init__(self, timeline, initially_running: bool, show_ui: bool, log_callback: Callable[[str], None]):
        self.timeline = timeline
        self.running = initially_running
        self._log_callback = log_callback
        self.window = None
        self.status_label = None
        self.toggle_button = None

        if not show_ui:
            return

        try:
            import omni.ui as ui

            self.window = ui.Window("Franka Pick and Place", width=300, height=120)
            with self.window.frame:
                with ui.VStack(spacing=8, height=0):
                    ui.Label("Execution control", height=24)
                    self.status_label = ui.Label("")
                    self.toggle_button = ui.Button("", height=34, clicked_fn=self.toggle)
            self._update_widgets()
        except Exception as exc:
            self._log_callback(f"UI controls unavailable ({exc}); starting automatically")
            self.running = True

    def _update_widgets(self):
        if self.status_label is None or self.toggle_button is None:
            return
        if self.running:
            self.status_label.text = "Status: Running"
            self.toggle_button.text = "Pause Pick/Place"
        else:
            self.status_label.text = "Status: Paused"
            self.toggle_button.text = "Start Pick/Place"

    def toggle(self):
        self.running = not self.running
        if self.running:
            self.timeline.play()
            self._log_callback("Pick-and-place execution resumed")
        else:
            self.timeline.pause()
            self._log_callback("Pick-and-place execution paused")
        self._update_widgets()

    def close(self):
        if self.window is not None:
            self.window.visible = False
            self.window = None


class VideoRecorder:
    """Record the active Isaac viewport to an MP4 for each completed run."""

    def __init__(self, requested: bool, output_dir: Path, log_callback: Callable[[str], None]):
        self.requested = requested
        self.output_dir = output_dir
        self._log_callback = log_callback
        self._capture = None
        self._viewport = None
        self._active = False
        self._last_output: Optional[str] = None

    @property
    def available(self) -> bool:
        return self._capture is not None and self._viewport is not None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def done(self) -> bool:
        return self._active and bool(self._capture.done)

    def initialize(self) -> None:
        if not self.requested:
            return
        try:
            from omni.kit.capture.viewport import CaptureExtension
            import omni.kit.viewport.utility as viewport_utils

            self._capture = CaptureExtension.get_instance()
            self._viewport = viewport_utils.get_active_viewport()
            if self._capture is None:
                raise RuntimeError("omni.kit.capture.viewport did not create a CaptureExtension instance")
            if self._viewport is None:
                raise RuntimeError("no active viewport; use --webrtc when recording video")

            # Make the evidence video use the same framing on every run.
            viewport_utils.frame_viewport_prims(self._viewport, ["/World/robot", "/World/Cube"])
            self._log_callback(
                f"VIDEO_READY camera={self._viewport.camera_path.pathString} resolution={self._viewport.resolution}"
            )
        except Exception as exc:
            self._capture = None
            self._viewport = None
            self._log_callback(f"VIDEO_UNAVAILABLE reason={exc}")

    def start(self, run_number: int, advance_callback: Callable[[float], bool]) -> bool:
        if not self.available:
            return False

        try:
            from omni.kit.capture.viewport import CaptureOptions, CaptureRangeType, CaptureRenderPreset

            options = CaptureOptions()
            options.camera = self._viewport.camera_path.pathString
            options.output_folder = str(self.output_dir)
            options.file_name = f"run-{run_number:03d}"
            options.file_type = ".mp4"
            options.range_type = CaptureRangeType.FRAMES
            options.start_frame = 1
            # The callback ends the capture at the actual run boundary. This
            # upper bound only keeps CaptureOptions valid for a long run.
            options.end_frame = 2000
            options.capture_every_Nth_frames = 1
            options.fps = 60
            options.res_width = 1280
            options.res_height = 720
            options.render_preset = CaptureRenderPreset.RAY_TRACE
            options.overwrite_existing_frames = True
            self._capture.options = options
            self._capture.show_default_progress_window = False
            self._capture.forward_one_frame_fn = advance_callback
            self._last_output = None
            self._active = bool(self._capture.start())
            if not self._active:
                self._log_callback(f"VIDEO_START_FAILED run={run_number}")
                return False
            self._log_callback(f"VIDEO_START run={run_number} path={options.get_full_path()}")
            return True
        except Exception as exc:
            self._active = False
            self._log_callback(f"VIDEO_START_FAILED run={run_number} reason={exc}")
            return False

    def finish(self, run_number: int) -> Optional[str]:
        if not self._active:
            return self._last_output
        try:
            outputs = self._capture.get_outputs()
            if outputs:
                self._last_output = outputs[0]
            self._log_callback(f"VIDEO_DONE run={run_number} outputs={outputs}")
        except Exception as exc:
            self._log_callback(f"VIDEO_FINISH_FAILED run={run_number} reason={exc}")
        finally:
            self._capture.forward_one_frame_fn = None
            self._active = False
        return self._last_output


def main():
    global execution_control

    logger.log("Starting Simple Franka Pick-and-Place Demo")
    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    pick_place = FrankaPickPlace(
        reset_settle_steps=args.reset_settle_steps,
        fault_open_gripper=args.fault_open_gripper or fault1_variant in {"injected-fault", "wrong-cause"},
        fault_zero_cube_friction=args.fault_zero_cube_friction or fault2_variant in {"injected-fault", "wrong-cause"},
        fault_high_grasp_approach=args.fault_high_grasp_approach or fault3_variant in {"injected-fault", "wrong-cause"},
        fault1_variant=fault1_variant,
        fault2_variant=fault2_variant,
        fault3_variant=fault3_variant,
        midway_drop_cause=args.midway_drop_cause,
        midway_drop_variant=args.midway_drop_variant,
        misplacement_cause=args.misplacement_cause,
        misplacement_variant=args.misplacement_variant,
        log_callback=logger.log,
    )
    cube_initial_position = (
        None if args.cube_initial_position is None else np.asarray(args.cube_initial_position, dtype=float)
    )
    pick_place.setup_scene(cube_initial_position=cube_initial_position)

    # Let USD references, articulation tensors, and the viewport finish their
    # initial update before the first reset is measured.
    for _ in range(10):
        simulation_app.update()

    video_recorder = VideoRecorder(args.record_video, session_dir, logger.log)
    video_recorder.initialize()

    timeline = omni.timeline.get_timeline_interface()
    show_controls = args.webrtc or not args.headless
    initially_running = args.auto_start or args.test or not show_controls
    execution_control = ExecutionControl(timeline, initially_running, show_controls, logger.log)
    # Start the timeline before entering the run loop.  CaptureExtension may
    # temporarily pause it while encoding, but it still needs a running
    # timeline to invoke ``forward_one_frame_fn`` for the captured frames.
    if execution_control.running:
        timeline.play()
    else:
        timeline.pause()
    simulation_app.update()

    logger.log("Starting pick-and-place execution")
    completed_runs = 0
    reset_needed = True
    current_run: Optional[Dict[str, Any]] = None
    stop_requested = False

    def advance_one_step(_dt: float = 0.0, manual_physics_step: bool = False) -> bool:
        """Advance reset settling, task motion, or post-release settling.

        Viewport capture disables timeline auto-update while it renders a
        sequence.  In that mode the callback must explicitly advance PhysX;
        the normal application loop continues to use ``simulation_app.update``
        for the physics step.
        """
        if current_run is None:
            return False

        if not pick_place.is_done():
            pick_place.forward(args.ik_method)
            if manual_physics_step:
                SimulationManager.step()
            current_run["steps"] += 1
            return True

        if current_run["post_settle_remaining"] > 0:
            pick_place.observe(label="post_run_settle")
            if manual_physics_step:
                SimulationManager.step()
            current_run["post_settle_remaining"] -= 1
            current_run["steps"] += 1
            return True

        if current_run["task_done_monotonic"] is None:
            current_run["task_done_monotonic"] = time.monotonic()
        return False

    def start_run() -> None:
        nonlocal current_run
        run_number = completed_runs + 1
        # A completed capture may leave the timeline playing while the next
        # reset is applied. Pause it first so gravity cannot move the cube
        # between the reset write and the first captured frame.
        if video_recorder.available:
            timeline.pause()
        pick_place.reset()
        current_run = {
            "run": run_number,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "started_monotonic": time.monotonic(),
            "task_done_monotonic": None,
            "steps": 0,
            "post_settle_remaining": args.post_run_settle_steps,
            "video_active": False,
        }
        if args.repeat == 0:
            logger.log(f"Starting pick-and-place run {run_number} (infinite mode)")
        else:
            logger.log(f"Starting pick-and-place run {run_number}/{args.repeat}")

        if video_recorder.available:
            current_run["video_active"] = video_recorder.start(
                run_number,
                lambda dt: advance_one_step(dt, manual_physics_step=True),
            )
            if not current_run["video_active"]:
                timeline.play()

    def finish_run() -> None:
        nonlocal completed_runs, current_run, reset_needed, stop_requested
        if current_run is None:
            return

        run_number = int(current_run["run"])
        if current_run["video_active"]:
            video_path = video_recorder.finish(run_number)
        else:
            video_path = None

        metrics = pick_place.evaluate_success()
        finished_monotonic = time.monotonic()
        task_done_monotonic = current_run["task_done_monotonic"] or finished_monotonic
        run_record = {
            "run": run_number,
            "fault1_variant": fault1_variant,
            "fault2_variant": fault2_variant,
            "fault3_variant": fault3_variant,
            "midway_drop_cause": args.midway_drop_cause,
            "midway_drop_variant": args.midway_drop_variant,
            "misplacement_cause": args.misplacement_cause,
            "misplacement_variant": args.misplacement_variant,
            "fault": fault_name,
            "started_at": current_run["started_at"],
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "task_elapsed_wall_seconds": task_done_monotonic - current_run["started_monotonic"],
            "total_elapsed_wall_seconds": finished_monotonic - current_run["started_monotonic"],
            "simulation_steps": current_run["steps"],
            "simulation_seconds_at_60hz": current_run["steps"] / 60.0,
            "video": video_path,
            "phase_history": pick_place.get_phase_history(),
            "metrics": metrics,
        }
        run_dir = session_dir / f"run-{run_number:03d}"
        write_json(run_dir / "result.json", run_record)
        write_trace(run_dir / "trace.jsonl", pick_place.get_trace())
        completed_runs += 1
        logger.log(
            "RUN_RESULT "
            f"run={run_number} success={metrics['success']} "
            f"task_elapsed_s={run_record['task_elapsed_wall_seconds']:.3f} "
            f"steps={current_run['steps']} "
            f"max_lift_m={metrics['max_lift_m']:.4f} "
            f"final_xy_error_m={metrics['final_xy_error_m']:.4f} "
            f"reasons={metrics['failure_reasons']}"
        )
        current_run = None
        reset_needed = True
        if args.test or (args.repeat > 0 and completed_runs >= args.repeat):
            stop_requested = True

    while simulation_app.is_running() and not stop_requested:
        # In video mode CaptureExtension calls advance_one_step from its
        # update callback. In normal mode the main loop drives the same method.
        if execution_control.running and SimulationManager.is_simulating():
            if reset_needed and current_run is None:
                start_run()

            if current_run is not None and not current_run["video_active"]:
                if not advance_one_step():
                    finish_run()

            elif current_run is not None and current_run["video_active"] and video_recorder.done:
                finish_run()

        simulation_app.update()

        # CaptureExtension stops the timeline while encoding. Check completion
        # after update even if SimulationManager no longer reports simulating.
        if current_run is not None and current_run["video_active"] and video_recorder.done:
            finish_run()

    summary = {
        "session": session_name,
        "config": {
            "seed": args.seed,
            "start_condition_id": args.start_condition_id,
            "repeat": args.repeat,
            "device": args.device,
            "ik_method": args.ik_method,
            "reset_settle_steps": args.reset_settle_steps,
            "post_run_settle_steps": args.post_run_settle_steps,
            "record_video": args.record_video,
            "fault1_variant": fault1_variant,
            "fault2_variant": fault2_variant,
            "fault3_variant": fault3_variant,
            "midway_drop_cause": args.midway_drop_cause,
            "midway_drop_variant": args.midway_drop_variant,
            "misplacement_cause": args.misplacement_cause,
            "misplacement_variant": args.misplacement_variant,
            "task_contract": {
                "cube_initial_position_m": pick_place.cube_initial_position.tolist(),
                "cube_size_m": pick_place.cube_size.tolist(),
                "target_position_m": pick_place.target_position.tolist(),
                "events_dt_steps": list(pick_place.events_dt),
                "xy_tolerance_m": 0.08,
                "z_tolerance_m": 0.06,
            },
            "fault": {
                "active": fault_name is not None,
                "name": fault_name,
                **fault_details,
            },
        },
        "completed_runs": completed_runs,
        "runs": [
            json.loads((session_dir / f"run-{index:03d}" / "result.json").read_text(encoding="utf-8"))
            for index in range(1, completed_runs + 1)
        ],
    }
    write_json(session_dir / "summary.json", summary)
    logger.log(f"SESSION_DONE completed_runs={completed_runs} summary={session_dir / 'summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.log(f"ERROR {exc}")
        logger.log("TRACEBACK " + __import__("traceback").format_exc().replace("\n", "\\n"))
    finally:
        if execution_control is not None:
            execution_control.close()
        kit_log = copy_latest_kit_log(session_dir)
        if kit_log is not None:
            logger.log(f"KIT_LOG {kit_log}")
        simulation_app.close()
        logger.close()
