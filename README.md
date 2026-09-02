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
