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
```

The scene creates a Franka Panda arm, a blue cube, a ground environment, and a
seven-phase state machine: move above the cube, approach, grasp, lift, move to
the target, release, and retract.

The official source is available at:
<https://github.com/isaac-sim/IsaacSim/tree/v5.1.0/source/standalone_examples/api/isaacsim.robot.manipulators/franka>
