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
args, _ = parser.parse_known_args()
if args.repeat < 0:
    parser.error("--repeat must be 0 or a positive integer")

from isaacsim import SimulationApp

simulation_config = {"headless": args.headless or args.webrtc}
if args.webrtc:
    # WebRTC needs a headless renderer while keeping the streamed UI visible.
    simulation_config["hide_ui"] = False

simulation_app = SimulationApp(simulation_config)

if args.webrtc:
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.services.livestream.nvcf")
    simulation_app.update()

import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from examples.franka_pick_place.pick_place import FrankaPickPlace

execution_control = None


class ExecutionControl:
    """Small Isaac Sim UI window for starting and pausing the state machine."""

    def __init__(self, timeline, initially_running: bool, show_ui: bool):
        self.timeline = timeline
        self.running = initially_running
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
            # A headless/minimal Kit build may not have omni.ui available.
            # Keep the simulation usable by falling back to automatic start.
            print(f"UI controls unavailable ({exc}); starting automatically")
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
            print("Pick-and-place execution resumed")
        else:
            self.timeline.pause()
            print("Pick-and-place execution paused")
        self._update_widgets()

    def close(self):
        if self.window is not None:
            self.window.visible = False
            self.window = None


def main():
    global execution_control

    print("Starting Simple Franka Pick-and-Place Demo")
    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    pick_place = FrankaPickPlace()
    pick_place.setup_scene()

    timeline = omni.timeline.get_timeline_interface()
    show_controls = args.webrtc or not args.headless
    initially_running = args.auto_start or args.test or not show_controls
    execution_control = ExecutionControl(timeline, initially_running, show_controls)
    if execution_control.running:
        timeline.play()
    else:
        timeline.pause()
    simulation_app.update()

    reset_needed = True
    completed_runs = 0

    print("Starting pick-and-place execution")
    while simulation_app.is_running():
        if execution_control.running and SimulationManager.is_simulating():
            if reset_needed:
                pick_place.reset()
                reset_needed = False
                run_number = completed_runs + 1
                if args.repeat == 0:
                    print(f"Starting pick-and-place run {run_number} (infinite mode)")
                else:
                    print(f"Starting pick-and-place run {run_number}/{args.repeat}")

            # Execute one step of the pick-and-place operation
            pick_place.forward(args.ik_method)

            if pick_place.is_done():
                completed_runs += 1
                print(f"Done picking and placing (run {completed_runs})")

                # --test always stops after one run; otherwise --repeat controls
                # whether another reset-and-run cycle should start.
                if args.test or (args.repeat > 0 and completed_runs >= args.repeat):
                    break
                reset_needed = True

        simulation_app.update()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if execution_control is not None:
            execution_control.close()
        simulation_app.close()
