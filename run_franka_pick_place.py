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


def main():
    print("Starting Simple Franka Pick-and-Place Demo")
    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    pick_place = FrankaPickPlace()
    pick_place.setup_scene()

    # Play the simulation.
    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    reset_needed = True
    completed_runs = 0

    print("Starting pick-and-place execution")
    while simulation_app.is_running():
        if SimulationManager.is_simulating():
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
        simulation_app.close()
