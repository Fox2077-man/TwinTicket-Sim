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
parser.add_argument(
    "--fault-open-gripper",
    action="store_true",
    help="Inject the first controlled fault: keep the gripper open during the grasp phase",
)
args, _ = parser.parse_known_args()
if args.repeat < 0:
    parser.error("--repeat must be 0 or a positive integer")
if args.reset_settle_steps < 0 or args.post_run_settle_steps < 0:
    parser.error("settle step counts must be non-negative")

random.seed(args.seed)
np.random.seed(args.seed)


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
fault_name = "open-gripper-during-grasp" if args.fault_open_gripper else None
logger.log(
    f"SESSION_START dir={session_dir} seed={args.seed} repeat={args.repeat} "
    f"device={args.device} fault={fault_name or 'none'}"
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
        fault_open_gripper=args.fault_open_gripper,
        log_callback=logger.log,
    )
    pick_place.setup_scene()

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
            "repeat": args.repeat,
            "device": args.device,
            "ik_method": args.ik_method,
            "reset_settle_steps": args.reset_settle_steps,
            "post_run_settle_steps": args.post_run_settle_steps,
            "record_video": args.record_video,
            "fault": {
                "active": args.fault_open_gripper,
                "name": fault_name,
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
