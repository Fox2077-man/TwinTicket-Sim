"""Run the remaining start-condition batches with stop gates and resume support."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from audit_calibration_dataset import sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, default=Path("calibration_scaleup_plan.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration-300"))
    parser.add_argument("--sim-python", default="/isaac-sim/python.sh")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


class ScaleupRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.repo_root.resolve()
        self.plan_path = resolve(self.root, args.plan)
        self.output = resolve(self.root, args.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output / "scaleup-execution.log"
        self.state_path = self.output / "scaleup-state.json"
        self.plan = read_json(self.plan_path)
        self.state = self.load_state()

    def log(self, message: str) -> None:
        line = f"{now()} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def load_state(self) -> Dict[str, Any]:
        plan_hash = sha256(self.plan_path)
        if self.state_path.is_file():
            state = read_json(self.state_path)
            if state.get("plan_sha256") != plan_hash:
                raise RuntimeError("scale-up plan changed after execution state was created")
            return state
        state = {
            "schema": "calibration-scaleup-execution-v1",
            "created_at": now(),
            "plan_path": str(self.plan_path.relative_to(self.root)),
            "plan_sha256": plan_hash,
            "conditions": {},
            "status": "running",
        }
        write_json(self.state_path, state)
        return state

    def save_state(self) -> None:
        self.state["updated_at"] = now()
        write_json(self.state_path, self.state)

    def validate_plan(self) -> None:
        conditions = self.plan.get("conditions") or []
        ids = [item.get("id") for item in conditions]
        positions = [tuple(item.get("cube_initial_position_m") or []) for item in conditions]
        datasets = [item.get("dataset_dir") for item in conditions]
        if len(conditions) != 10 or len(set(ids)) != 10:
            raise RuntimeError("plan must contain exactly ten unique condition IDs")
        if len(set(positions)) != 10:
            raise RuntimeError("plan contains duplicate start positions")
        if len(set(datasets)) != 10:
            raise RuntimeError("plan contains duplicate dataset directories")
        for condition in conditions:
            if len(condition.get("cube_initial_position_m") or []) != 3:
                raise RuntimeError(f"invalid position for {condition.get('id')}")
            dataset = resolve(self.root, condition["dataset_dir"])
            state_path = dataset / "batch-state.json"
            if condition["source"] == "generated" and dataset.exists() and not state_path.is_file():
                if any(dataset.iterdir()):
                    raise RuntimeError(
                        f"refusing to reuse nonempty untracked batch directory: {dataset}"
                    )
            if state_path.is_file():
                batch = read_json(state_path)
                actual = batch.get("start_condition") or {}
                if actual.get("id") != condition["id"] or actual.get("cube_initial_position_m") != condition["cube_initial_position_m"]:
                    raise RuntimeError(
                        f"condition ID/coordinate conflict in existing batch: {dataset}"
                    )

    def command(self, command: List[str], label: str, allowed: tuple[int, ...] = (0,)) -> int:
        self.log(f"COMMAND_START label={label} command={json.dumps(command)}")
        with self.log_path.open("a", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                cwd=self.root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.log(f"COMMAND_DONE label={label} returncode={completed.returncode}")
        if completed.returncode not in allowed:
            self.log(f"COMMAND_BLOCKED label={label} returncode={completed.returncode}")
        return completed.returncode

    def audit_default(self, condition: Mapping[str, Any]) -> bool:
        command = [
            sys.executable,
            str(self.root / "audit_calibration_dataset.py"),
            "--repo-root",
            str(self.root),
            "--output-dir",
            str(resolve(self.root, condition["dataset_dir"])),
            "--verify",
        ]
        return self.command(command, f"verify:{condition['id']}") == 0

    def audit_batch(self, condition: Mapping[str, Any], mode: str) -> bool:
        command = [
            sys.executable,
            str(self.root / "audit_start_condition_batch.py"),
            "--repo-root",
            str(self.root),
            "--batch-dir",
            str(resolve(self.root, condition["dataset_dir"])),
            f"--{mode}",
        ]
        return self.command(command, f"{mode}:{condition['id']}") == 0

    def aggregate_audit(self, mode: str, incomplete_ok: bool = False) -> bool:
        command = [
            sys.executable,
            str(self.root / "audit_calibration_scaleup.py"),
            "--repo-root",
            str(self.root),
            "--plan",
            str(self.plan_path),
            "--output-dir",
            str(self.output),
            f"--{mode}",
        ]
        allowed = (0, 1) if incomplete_ok else (0,)
        return self.command(command, f"aggregate:{mode}", allowed) == 0

    def condition_state(self, condition: Mapping[str, Any], status: str, detail: str) -> None:
        self.state["conditions"][condition["id"]] = {
            "status": status,
            "detail": detail,
            "dataset_dir": condition["dataset_dir"],
            "checked_at": now(),
        }
        self.save_state()

    def run_generated(self, condition: Mapping[str, Any]) -> bool:
        dataset = resolve(self.root, condition["dataset_dir"])
        batch_state = dataset / "batch-state.json"
        reference = dataset / "reference-manifest.json"
        if batch_state.is_file():
            state = read_json(batch_state)
            if (state.get("counts") or {}).get("accepted_records") == 30:
                mode = "verify" if reference.is_file() else "initialize"
                if not self.audit_batch(condition, mode):
                    self.condition_state(condition, "blocked", f"completed batch failed {mode} audit")
                    return False
                if mode == "initialize" and not self.audit_batch(condition, "verify"):
                    self.condition_state(condition, "blocked", "new reference failed verification")
                    return False
                self.condition_state(condition, "passed", "existing completed batch reverified")
                return True

        command = [
            sys.executable,
            str(self.root / "run_start_condition_batch.py"),
            "--results-dir",
            str(dataset),
            "--sim-python",
            self.args.sim_python,
            "--run-script",
            str(self.root / "run_franka_pick_place.py"),
            "--device",
            self.args.device,
            "--max-attempts",
            str(self.args.max_attempts),
            "--start-condition-id",
            condition["id"],
            "--cube-initial-position",
            *(str(value) for value in condition["cube_initial_position_m"]),
        ]
        if batch_state.is_file():
            command.append("--resume")
        try:
            returncode = self.command(command, f"run:{condition['id']}")
        except KeyboardInterrupt:
            self.log(f"INTERRUPTED condition={condition['id']}; reconciling partial attempt")
            reconcile = command + ["--reconcile-only"]
            if "--resume" not in reconcile:
                reconcile.append("--resume")
            self.command(reconcile, f"reconcile:{condition['id']}", (0, 1, 2))
            self.audit_batch(condition, "status")
            self.condition_state(condition, "interrupted", "partial attempt retained and reconciled")
            raise
        if returncode != 0:
            self.audit_batch(condition, "status")
            self.condition_state(
                condition,
                "blocked",
                "batch stopped at the first slot that exhausted its expected-outcome gate",
            )
            return False
        if not self.audit_batch(condition, "initialize"):
            self.condition_state(condition, "blocked", "completed batch failed reference initialization")
            return False
        if not self.audit_batch(condition, "verify"):
            self.condition_state(condition, "blocked", "completed batch failed frozen-reference verification")
            return False
        self.condition_state(condition, "passed", "30 records generated and quality verified")
        return True

    def run(self) -> int:
        self.validate_plan()
        if self.args.status_only:
            self.aggregate_audit("status", incomplete_ok=True)
            return 0
        self.state["status"] = "running"
        self.state.pop("blocked_condition", None)
        self.state.pop("finished_at", None)
        self.save_state()
        for condition in self.plan.get("conditions", []):
            self.log(f"CONDITION_START id={condition['id']} source={condition['source']}")
            if condition["source"] == "canonical-existing":
                passed = self.audit_default(condition)
                detail = "canonical 30 reverified" if passed else "canonical quality verification failed"
            elif condition["source"] == "validated-existing":
                passed = self.audit_batch(condition, "verify")
                detail = "validated 30 reverified" if passed else "validated batch verification failed"
            else:
                passed = self.run_generated(condition)
                detail = "generated batch passed" if passed else "generated batch blocked"
            if condition["source"] != "generated":
                self.condition_state(condition, "passed" if passed else "blocked", detail)
            self.aggregate_audit("status", incomplete_ok=True)
            if not passed:
                self.state["status"] = "blocked"
                self.state["blocked_condition"] = condition["id"]
                self.state["finished_at"] = now()
                self.save_state()
                self.log(f"SCALEUP_BLOCKED condition={condition['id']}; later conditions were not run")
                return 1
            self.log(f"CONDITION_DONE id={condition['id']} status=passed")

        aggregate_reference = self.output / "aggregate-reference-manifest.json"
        if aggregate_reference.is_file():
            final_ok = self.aggregate_audit("verify")
        else:
            final_ok = self.aggregate_audit("initialize") and self.aggregate_audit("verify")
        self.state["status"] = "complete" if final_ok else "blocked"
        self.state["finished_at"] = now()
        if not final_ok:
            self.state["blocked_condition"] = "aggregate-quality"
        else:
            self.state.pop("blocked_condition", None)
        self.save_state()
        self.log(f"SCALEUP_DONE status={self.state['status']}")
        return 0 if final_ok else 1


def main() -> int:
    args = parse_args()
    output = resolve(args.repo_root.resolve(), args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".scaleup.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Another scale-up process holds {lock_path}", file=sys.stderr)
            return 2
        lock.seek(0)
        lock.truncate()
        lock.write(f"pid={os.getpid()} started_at={now()}\n")
        lock.flush()
        runner = ScaleupRunner(args)
        try:
            return runner.run()
        except KeyboardInterrupt:
            runner.state["status"] = "paused"
            runner.state["paused_at"] = now()
            runner.state.pop("blocked_condition", None)
            runner.state.pop("finished_at", None)
            runner.save_state()
            return 130
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"Scale-up stopped safely: {exc}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
