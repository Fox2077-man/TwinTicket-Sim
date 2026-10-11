# TwinTicket-Sim

This repository contains a standalone Franka Panda pick-and-place example for
Isaac Sim 5.1. The implementation is based on NVIDIA's Isaac Sim `v5.1.0`
example and keeps the original Apache-2.0 headers in each copied source file.

## Run

Run the example with the Isaac Sim Python interpreter from the repository root:

```bash
python run_franka_pick_place.py
```

Useful options:

```bash
python run_franka_pick_place.py --headless --test
python run_franka_pick_place.py --device cuda --ik-method pseudoinverse
python run_franka_pick_place.py --repeat 5
python run_franka_pick_place.py --webrtc --repeat 0
python run_franka_pick_place.py --headless --auto-start --repeat 0
python run_franka_pick_place.py --headless --auto-start --repeat 5 \
  --device cpu --seed 0 --results-dir artifacts/baseline
```

`--repeat N` runs the complete sequence `N` times and resets the robot and cube
between runs. `--repeat 0` repeats indefinitely. The default is one run, and
`--test` always exits after the first completed run.

Each run resets the cube to its fixed default tabletop position
`[0.5, 0.0, 0.0258]` (meters), including between repeated runs.

Visible and WebRTC sessions open a **Franka Pick and Place** window. Click
**Start Pick/Place** to begin and **Pause Pick/Place** to freeze the simulation;
click it again to continue from the same phase. Interactive sessions initially
wait for this button. Add `--auto-start` if execution should begin immediately.

To view and control the running demo through the Isaac Sim WebRTC Streaming
Client, use `--webrtc` and pass the host's reachable IP address as a Kit
setting:

```bash
python run_franka_pick_place.py --webrtc \
  --/app/livestream/publicEndpointAddress=HOST_IP \
  --/app/livestream/port=49100
```

The host must allow TCP `49100` and UDP `47998`. Connect the WebRTC client to
the same `HOST_IP`. Do not use `--test` for an interactive session because it
closes the application as soon as the motion sequence completes.

When using the NVIDIA Docker image, run this from the host terminal. Stop any
other Isaac Sim container first so that the WebRTC ports are available:

```bash
sudo docker stop isaac-sim
HOST_IP=117.50.196.163
sudo docker compose -f /home/ubuntu/docker-compose.yml run --rm --no-deps \
  --entrypoint /bin/bash \
  -v /home/ubuntu/TwinTicket-Sim:/workspace/TwinTicket-Sim \
  -w /workspace/TwinTicket-Sim \
  isaac-sim \
  -lc "/isaac-sim/python.sh run_franka_pick_place.py --webrtc --repeat 0 \
    --/persistent/isaac/asset_root/default=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1 \
    --/app/livestream/publicEndpointAddress=${HOST_IP} \
    --/app/livestream/port=49100"
```

Replace `HOST_IP` with the address reachable from the computer running the
WebRTC client. For an Internet connection, use the server's public IP and
allow TCP `49100` plus UDP `47998` in the firewall or cloud security group.

The scene creates a Franka Panda arm, a blue cube, a ground environment, and a
seven-phase state machine: move above the cube, approach, grasp, lift, move to
the target, release, and retract.

## Repeatable baseline experiments

The runner fixes the cube reset pose, robot default pose, phase targets, and
random seeds. `--reset-settle-steps` holds the reset pose before motion and
`--post-run-settle-steps` lets the released cube settle before success is
evaluated. The default values are 15 and 30 frames respectively. Keep these
values, the IK method, and the simulation device unchanged when comparing a
fault-injected run with a normal baseline.

Each invocation creates a timestamped directory below `--results-dir`:

```text
session-YYYYMMDD-HHMMSS/
  experiment.log       # line-buffered lifecycle and phase log
  kit.log              # Isaac Sim Kit log copied before shutdown
  summary.json         # configuration and all run results
  run-001/
    result.json        # success, timings, metrics, phase history
    trace.jsonl         # per-step cube/EE/DOF observations
  run-001.mp4          # when --record-video is enabled
```

Use `--record-video` with `--webrtc` to save one 1280x720 MP4 for every run:

```bash
python run_franka_pick_place.py --webrtc --record-video --auto-start \
  --repeat 3 --device cpu --seed 0 --results-dir artifacts/video-baseline
```

Video capture also leaves the numbered PNG frames beside each MP4. Recording
is slower because every frame is rendered and encoded; compare simulation
steps and physical metrics in `result.json` rather than wall-clock time alone.

Success requires all of the following: the seven-phase state machine completed,
the cube rose by at least 0.05 m, the final XY position is within 0.08 m of the
target, and the final cube height is within 0.06 m of the tabletop. This avoids
counting a motion sequence that merely reached its last phase as a physical
pick-and-place success.

## First controlled fault: open gripper during grasp

`--fault-open-gripper` changes exactly one commanded behavior: during the grasp
phase the fingers remain at their fully open position instead of closing. The
robot path, phase durations, cube pose, target, physics settings, and random
seed are unchanged. The option is off by default, so removing it restores the
normal behavior.

Run a matched normal/fault pair by keeping every option except the fault switch
identical:

```bash
python run_franka_pick_place.py --webrtc --record-video --auto-start --repeat 1 \
  --device cpu --seed 0 --results-dir artifacts/controlled-fault-01/normal
python run_franka_pick_place.py --webrtc --record-video --auto-start --repeat 1 \
  --device cpu --seed 0 --fault-open-gripper \
  --results-dir artifacts/controlled-fault-01/fault
```

The selected fault is recorded in `summary.json`, and the faulted phase emits a
`FAULT_INJECTED` entry in `experiment.log`. Per-step finger positions remain
available in `trace.jsonl` as DOFs 7 and 8.

## Second controlled fault: zero cube friction

`--fault-zero-cube-friction` keeps the normal robot path and normal gripper
close command, but binds a zero-friction physics material to the cube collision
surface. Its static and dynamic friction are both `0.0`, and the `min` combine
mode makes the resulting finger-cube friction zero. Without tangential contact
force, the gripper cannot carry the cube upward. Removing the option restores
the default cube contact material.

The three fault switches are mutually exclusive, so they cannot accidentally be
active in the same run:

```bash
python run_franka_pick_place.py --webrtc --record-video --auto-start --repeat 1 \
  --device cpu --seed 0 --fault-zero-cube-friction \
  --results-dir artifacts/controlled-fault-02/zero-friction
```

Both faults should visibly leave the cube near its starting position. Their
records distinguish the causes: the first fault has `grasp_command: open` and
open finger DOFs, while the second has `grasp_command: close`, closed finger
DOFs, and `contact_condition: zero_cube_friction`.

## Third controlled fault: high grasp approach

`--fault-high-grasp-approach` changes only the motion geometry: during the
approach phase the end effector targets a point `0.08 m` higher than the normal
grasp pose. The cube pose, robot reset, normal `close_gripper` command, and
default contact material are unchanged, so the fingers close above the cube
without making a geometric grasp. Removing the option restores the normal
approach height.

All three fault switches are mutually exclusive. A third-fault run can be
recorded with:

```bash
python run_franka_pick_place.py --webrtc --record-video --auto-start --repeat 1 \
  --device cpu --seed 0 --fault-high-grasp-approach \
  --results-dir artifacts/controlled-fault-03/high-grasp-approach
```

The third-fault record identifies `cause_path: approach_geometry`,
`grasp_command: close`, `contact_condition: default`,
`grasp_height_offset_m: 0.08`, and the larger
`grasp_ee_distance_at_command_m`. This separates it from the open-command and
zero-friction faults without changing the cube's starting position.

## Fault stability acceptance

Before collecting formal data, each fault was run independently 10 times with
the matched baseline settings (`seed=0`, CPU, fixed cube pose, and 15/30 settle
frames). The normal configuration succeeded 10/10; open-gripper, zero-friction,
and high-grasp-approach each failed 10/10, completed all 325 simulation steps,
and produced no runtime error or incomplete trace. The detailed statistics and
session paths are recorded in
`artifacts/controlled-fault-stability/stability-report.md`.

This is a reset-repeatability check at one fixed seed, not a cross-seed
statistical robustness study.

When running in the NVIDIA Docker image, the container uses UID 1234. Make the
results directory writable before a host-mounted run if necessary:

```bash
sudo mkdir -p artifacts
sudo chmod -R a+rwX artifacts
```

Do not commit generated experiment data; the repository ignores `artifacts/*`
while retaining `artifacts/.gitkeep`.

The official source is available at:
<https://github.com/isaac-sim/IsaacSim/tree/v5.1.0/source/standalone_examples/api/isaacsim.robot.manipulators/franka>

## Fault 1 four-version causal comparison

To make the first fault (the gripper does not close) a fair causal test, run
the dedicated comparison driver with the Isaac Sim interpreter:

```bash
/isaac-sim/python.sh run_fault1_comparison.py \
  --results-dir artifacts/fault1-comparison
```

The driver launches four fresh Isaac Sim processes with the same seed, device,
IK method, reset pose, target, phase durations, and settle steps:

1. `normal`: the grasp phase sends `close_gripper`.
2. `injected-fault`: the grasp phase sends `open_gripper`.
3. `wrong-cause`: the open command is deliberately kept, while a plausible
   but incorrect contact-friction diagnosis raises cube friction. This is an
   error-handling treatment; it does not remove or replace the true command
   fault.
4. `true-fix`: only the command is restored to `close_gripper`; all task
   difficulty parameters remain the normal values.

Each variant has its own `session-*` directory containing `experiment.log`,
`summary.json`, `run-001/result.json`, `run-001/trace.jsonl`, and
`run-001.mp4` when viewport capture is available. The parent directory also
contains `comparison.json` and `comparison.md`. The report checks that the
task contract is identical, that wrong-cause handling preserved the open
command, and that the true fix changed only the command path. The expected
causal pattern is normal success, injected-fault failure, wrong-cause failure,
and true-fix success.

## Fault 2 and Fault 3 four-version causal comparisons

The same four-version protocol is available for the other controlled faults:

```bash
/isaac-sim/python.sh run_fault23_comparison.py --fault 2 \
  --results-dir artifacts/fault2-comparison
/isaac-sim/python.sh run_fault23_comparison.py --fault 3 \
  --results-dir artifacts/fault3-comparison
```

For Fault 2 (`zero-cube-friction`), `wrong-cause` repeats the normal close
command as a gripper-command diagnosis while leaving the cube static/dynamic
friction at `0.0` with `min` combine mode. `true-fix` restores the default cube
contact material only.

For Fault 3 (`high-grasp-approach`), `wrong-cause` applies a height-related
reference diagnosis (`verify_cube_top_height`) while retaining the true
`+0.08 m` approach height offset. It does not repeat the Fault 2 close command
and does not change the commanded grasp height. `true-fix` restores the normal
approach height only.

Each comparison writes four independent session directories with MP4 videos,
`experiment.log`, `summary.json`, `run-001/result.json`, and
`run-001/trace.jsonl`, plus `comparison.md` and `comparison.json`. The reports
check identical task contracts, matching pre-grasp trace prefixes, preserved
root causes in wrong-cause runs, and successful true-fix runs.

## Mid-transport drop comparison

The second symptom uses the same reproducible evidence layout. The cube is
grasped and lifted first; the selected intervention is applied at the start of
`move_to_target`, so each fault and wrong-cause run visibly drops the cube
while the true-fix run completes the placement:

```bash
/isaac-sim/python.sh run_midway_drop_comparison.py \
  --results-dir artifacts/midway-drop-comparison
```

The completed comparison is in `artifacts/midway-drop-comparison/`. It contains
one `injected-fault`, `wrong-cause`, and `true-fix` session for each cause:
`gripper-release` (opens the gripper), `friction-loss` (filters only the cube's
collisions with `panda_hand`, `panda_leftfinger`, and `panda_rightfinger`
during transport), and `motion-disturbance` (applies a downward velocity).
Each session has an MP4, `experiment.log`, `summary.json`,
`run-001/result.json`, and `run-001/trace.jsonl`; the parent
`comparison.md`/`comparison.json` records the fairness checks and causal
evidence. A run is marked unsuccessful whenever `drop_detected` is true,
even if the cube later settles inside the target tolerance.

## Inaccurate-placement comparison

The third symptom keeps the grasp and transport successful but places the cube
away from the nominal target. It compares three single-cause interventions:
`planner-x-offset` (the planned release target has only an X offset),
`transport-y-bias` (the planner target remains correct while only the transport
command has a Y bias), and `early-release` (the gripper opens at a fixed
pre-target transport step). Each cause has an `injected-fault`, `wrong-cause`,
and `true-fix` session:

```bash
/isaac-sim/python.sh run_misplacement_comparison.py \
  --results-dir artifacts/misplacement-comparison
```

Each record preserves the nominal target and records the planned release
position, commanded transport position, final XY error, and, where applicable,
the early-release step and distance remaining to the target. The report in
`artifacts/misplacement-comparison/comparison.md` checks matched control
prefixes, root-cause preservation in wrong-cause runs, and successful recovery
from the true fix.
