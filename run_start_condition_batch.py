"""Run one bounded 30-session batch under a new cube start condition.

The batch contains ten canonical records for each of the three symptoms. It
keeps every attempt, normally retries a slot at most once when evidence/quality
is invalid, and can append an explicitly requested repair attempt. Expected
fault outcomes are accepted calibration records, not void attempts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_START_CONDITION = {
    "id": "cube-x-minus-0.01-y-plus-0.02",
    "cube_initial_position_m": [0.49, 0.02, 0.0258],
    "changed_from_default": {"cube_x_m": -0.01, "cube_y_m": 0.02, "cube_z_m": 0.0},
}
VARIANTS = ("injected-fault", "wrong-cause", "true-fix")
REQUIRED = ("run-001.mp4", "run-001/result.json", "run-001/trace.jsonl", "experiment.log", "summary.json")
ERROR_PATTERN = re.compile(r"\b(ERROR|TRACEBACK|FATAL|Exception)\b")
SYMPTOMS = {
    "cannot-grasp": {
        "causes": (
            ("gripper-command-open", "--fault1-variant"),
            ("zero-cube-friction", "--fault2-variant"),
            ("high-grasp-approach", "--fault3-variant"),
        ),
    },
    "midway-drop": {
        "causes": tuple((cause, "--midway-drop-variant") for cause in ("gripper-release", "friction-loss", "motion-disturbance")),
    },
    "inaccurate-placement": {
        "causes": tuple((cause, "--misplacement-variant") for cause in ("planner-x-offset", "transport-y-bias", "early-release")),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("artifacts/start-condition-validation/cube-x-minus-0.01-y-plus-0.02"),
    )
    parser.add_argument("--sim-python", default=sys.executable)
    parser.add_argument("--run-script", type=Path, default=Path(__file__).with_name("run_franka_pick_place.py"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ik-method", default="damped-least-squares")
    parser.add_argument("--reset-settle-steps", type=int, default=15)
    parser.add_argument("--post-run-settle-steps", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--no-webrtc", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reconcile-only", action="store_true")
    parser.add_argument("--reject-reason", default=None)
    parser.add_argument("--start-condition-id", default=DEFAULT_START_CONDITION["id"])
    parser.add_argument(
        "--cube-initial-position",
        type=float,
        nargs=3,
        default=DEFAULT_START_CONDITION["cube_initial_position_m"],
        metavar=("X", "Y", "Z"),
    )
    parsed = parser.parse_args()
    if not 1 <= parsed.max_attempts <= 10:
        parser.error("--max-attempts must be between 1 and 10")
    return parsed


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_markdown_link(batch_root: Path, stored_path: str) -> str:
    """Render stored video paths as links relative to the batch index."""
    if not stored_path or stored_path == "n/a":
        return "n/a"
    stored = Path(stored_path)
    target = stored
    marker = "/artifacts/"
    if marker in stored_path:
        relative_to_repo = Path(stored_path.split(marker, 1)[1])
        repo_root = next(
            (parent.parent for parent in (batch_root, *batch_root.parents) if parent.name == "artifacts"),
            None,
        )
        if repo_root is not None:
            target = repo_root / "artifacts" / relative_to_repo
    try:
        relative = Path(os.path.relpath(target, batch_root)).as_posix()
    except ValueError:
        relative = stored_path
    return f"[打开视频]({relative})"


def slots() -> Iterable[Dict[str, str]]:
    for symptom, spec in SYMPTOMS.items():
        yield {"id": f"{symptom}/shared-normal/normal", "symptom": symptom, "cause": "shared-normal", "variant": "normal", "flag": ""}
        for cause, flag in spec["causes"]:
            for variant in VARIANTS:
                yield {"id": f"{symptom}/{cause}/{variant}", "symptom": symptom, "cause": cause, "variant": variant, "flag": flag}


def latest_session(directory: Path) -> Optional[Path]:
    sessions = [path for path in directory.glob("session-*") if path.is_dir()]
    return max(sessions, key=lambda path: path.stat().st_mtime) if sessions else None


def command_for(
    args: argparse.Namespace,
    slot: Dict[str, str],
    attempt_dir: Path,
    start_condition: Dict[str, Any],
) -> List[str]:
    command = [
        args.sim_python,
        str(args.run_script),
        "--headless",
        "--auto-start",
        "--test",
        "--repeat",
        "1",
        "--device",
        args.device,
        "--ik-method",
        args.ik_method,
        "--seed",
        str(args.seed),
        "--reset-settle-steps",
        str(args.reset_settle_steps),
        "--post-run-settle-steps",
        str(args.post_run_settle_steps),
        "--record-video",
        "--start-condition-id",
        start_condition["id"],
        "--cube-initial-position",
        *(str(value) for value in start_condition["cube_initial_position_m"]),
        "--results-dir",
        str(attempt_dir),
    ]
    if not args.no_webrtc:
        command.append("--webrtc")
    if slot["variant"] != "normal":
        if slot["symptom"] == "midway-drop":
            command.extend(["--midway-drop-cause", slot["cause"]])
        elif slot["symptom"] == "inaccurate-placement":
            command.extend(["--misplacement-cause", slot["cause"]])
        command.extend([slot["flag"], slot["variant"]])
    return command


def expected_outcome(slot: Dict[str, str], metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    expected_success = slot["variant"] in {"normal", "true-fix"}
    errors: List[str] = []
    if metrics.get("success") is not expected_success:
        errors.append(f"success_is_{metrics.get('success')}_expected_{expected_success}")
    if slot["symptom"] == "cannot-grasp" and not expected_success and metrics.get("lifted"):
        errors.append("fault_lifted_cube")
    if slot["symptom"] == "midway-drop":
        if not expected_success and not (metrics.get("grasp_established_before_fault") and metrics.get("drop_detected")):
            errors.append("missing_grasp_then_drop_evidence")
        if expected_success and metrics.get("drop_detected"):
            errors.append("recovery_still_dropped")
        expected_actions = {
            "gripper-release": "open_gripper_during_transport",
            "friction-loss": "disable_finger_cube_collision_during_transport",
            "motion-disturbance": "downward_disturbance_velocity",
        }
        if not expected_success and metrics.get("drop_action") != expected_actions[slot["cause"]]:
            errors.append("midway_drop_action_mismatch")
        if slot["cause"] == "motion-disturbance" and not expected_success:
            if metrics.get("drop_disturbance_velocity_z_mps") != -3.0:
                errors.append("motion_disturbance_magnitude_changed")
            if slot["variant"] == "wrong-cause":
                if metrics.get("midway_wrong_diagnosis") != "gripper_command":
                    errors.append("motion_wrong_diagnosis_mismatch")
                if metrics.get("midway_wrong_diagnosis_action") != "verify_close_command":
                    errors.append("motion_wrong_action_changed_physics")
    if slot["symptom"] == "inaccurate-placement":
        if not expected_success and not (metrics.get("lifted") and not metrics.get("drop_detected") and metrics.get("misplacement_detected")):
            errors.append("missing_grasp_transport_misplacement_evidence")
        if expected_success and metrics.get("misplacement_detected"):
            errors.append("recovery_still_misplaced")
    return not errors, errors


def inspect_attempt(
    root: Path,
    slot: Dict[str, str],
    attempt: int,
    returncode: int,
    command: List[str],
    start_condition: Dict[str, Any],
) -> Dict[str, Any]:
    attempt_dir = root / slot["symptom"] / slot["cause"] / slot["variant"] / f"attempt-{attempt:02d}"
    session = latest_session(attempt_dir)
    record: Dict[str, Any] = {
        "attempt": attempt,
        "attempt_directory": str(attempt_dir),
        "command": command,
        "returncode": returncode,
        "session": str(session) if session else None,
    }
    if session is None:
        record.update({"quality_passed": False, "quality_errors": ["session_missing"], "status": "void"})
        return record
    paths = {name: session / name for name in REQUIRED}
    errors = [f"missing:{name}" for name, path in paths.items() if not path.is_file()]
    summary: Dict[str, Any] = {}
    result: Dict[str, Any] = {}
    try:
        if paths["summary.json"].is_file():
            summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
        if paths["run-001/result.json"].is_file():
            result = json.loads(paths["run-001/result.json"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"json_parse:{exc}")
    config = summary.get("config") or {}
    metrics = result.get("metrics") or {}
    if config.get("start_condition_id") != start_condition["id"]:
        errors.append("start_condition_id_mismatch")
    actual_position = (config.get("task_contract") or {}).get("cube_initial_position_m")
    if actual_position != start_condition["cube_initial_position_m"]:
        errors.append(f"cube_initial_position_mismatch:{actual_position}")
    if paths["experiment.log"].is_file() and ERROR_PATTERN.search(paths["experiment.log"].read_text(encoding="utf-8")):
        errors.append("experiment_log_error_marker")
    outcome_valid, outcome_errors = expected_outcome(slot, metrics)
    errors.extend(outcome_errors)
    evidence_complete = all(path.is_file() for path in paths.values())
    quality_passed = returncode == 0 and evidence_complete and outcome_valid and not errors
    record.update(
        {
            "files": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in paths.items()
                if path.is_file()
            },
            "config": config,
            "metrics": metrics,
            "quality_passed": quality_passed,
            "quality_errors": errors,
            "status": "accepted" if quality_passed else "void",
        }
    )
    return record


def state_path(root: Path) -> Path:
    return root / "batch-state.json"


def load_state(root: Path, resume: bool, start_condition: Dict[str, Any]) -> Dict[str, Any]:
    path = state_path(root)
    if resume and path.is_file():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("start_condition") != start_condition:
            raise RuntimeError(
                "refusing to resume with a different start condition: "
                f"stored={state.get('start_condition')} requested={start_condition}"
            )
        return state
    if path.exists():
        raise RuntimeError(f"batch state already exists at {path}; use --resume")
    return {
        "schema": "start-condition-validation-batch-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_condition": start_condition,
        "planned_slot_count": 30,
        "slots": {},
    }


def write_reports(root: Path, state: Dict[str, Any]) -> None:
    accepted = [item for item in state["slots"].values() if item.get("accepted_attempt")]
    attempts = [attempt for item in state["slots"].values() for attempt in item.get("attempts", [])]
    successful_outcomes = sum(bool((item["accepted_attempt"].get("metrics") or {}).get("success")) for item in accepted)
    failed_outcomes = len(accepted) - successful_outcomes
    attempt_successes = sum((attempt.get("metrics") or {}).get("success") is True for attempt in attempts)
    attempt_failures = sum((attempt.get("metrics") or {}).get("success") is False for attempt in attempts)
    reruns = sum(max(0, len(item.get("attempts", [])) - 1) for item in state["slots"].values())
    void_attempts = sum(attempt.get("status") == "void" for attempt in attempts)
    counts = {
        "planned_slots": 30,
        "completed_slots": len(accepted),
        "accepted_records": len(accepted),
        "successful_outcomes": successful_outcomes,
        "failed_outcomes": failed_outcomes,
        "total_attempts": len(attempts),
        "attempt_successful_outcomes": attempt_successes,
        "attempt_failed_outcomes": attempt_failures,
        "attempts_without_outcome": len(attempts) - attempt_successes - attempt_failures,
        "rerun_attempts": reruns,
        "void_attempts": void_attempts,
    }
    state["counts"] = counts
    state["finished_at"] = datetime.now(timezone.utc).isoformat() if len(accepted) == 30 else None
    write_json(state_path(root), state)
    index = {
        "schema": "start-condition-validation-index-v1",
        "start_condition": state["start_condition"],
        "counts": counts,
        "records": [
            {
                "id": slot_id,
                "symptom": item["slot"]["symptom"],
                "cause": item["slot"]["cause"],
                "variant": item["slot"]["variant"],
                "session": item["accepted_attempt"]["session"],
                "attempt": item["accepted_attempt"]["attempt"],
            }
            for slot_id, item in state["slots"].items()
            if item.get("accepted_attempt")
        ],
    }
    write_json(root / "batch-index.json", index)
    lines = [
        "# 新开始条件小批次",
        "",
        f"开始条件：`{state['start_condition']['id']}`，方块初始位置 `{state['start_condition']['cube_initial_position_m']}`。除初始方块位置外，机器人、目标、控制器、种子和任务参数保持不变。",
        "",
        f"- 计划槽位：**{counts['planned_slots']}**；已接受：**{counts['accepted_records']}**。",
        f"- 成功结果：**{counts['successful_outcomes']}**；预期失败结果：**{counts['failed_outcomes']}**。",
        f"- 实际尝试：**{counts['total_attempts']}**；实际成功：**{counts['attempt_successful_outcomes']}**；实际失败：**{counts['attempt_failed_outcomes']}**；无结果：**{counts['attempts_without_outcome']}**。",
        f"- 重跑：**{counts['rerun_attempts']}**；作废尝试：**{counts['void_attempts']}**。",
        "",
        "| 数据 ID | 尝试 | 任务成功 | 状态 | 视频 |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for slot_id, item in state["slots"].items():
        accepted_attempt = item.get("accepted_attempt")
        if not accepted_attempt:
            lines.append(f"| `{slot_id}` | {len(item.get('attempts', []))} | n/a | 未接受 | n/a |")
            continue
        video = (accepted_attempt.get("files") or {}).get("run-001.mp4", {}).get("path", "n/a")
        success = (accepted_attempt.get("metrics") or {}).get("success")
        lines.append(
            f"| `{slot_id}` | {accepted_attempt['attempt']} | `{success}` | accepted | "
            f"{video_markdown_link(root, video)} |"
        )
    lines.extend([
        "",
        "## 全部实际尝试",
        "",
        "| 数据 ID | 尝试 | 实际结果 | 状态 | 视频 | 质量问题 |",
        "| --- | ---: | --- | --- | --- | --- |",
    ])
    for slot_id, item in state["slots"].items():
        for attempt in item.get("attempts", []):
            outcome = (attempt.get("metrics") or {}).get("success")
            errors = "; ".join(attempt.get("quality_errors") or []) or "none"
            video = (attempt.get("files") or {}).get("run-001.mp4", {}).get("path", "n/a")
            lines.append(
                f"| `{slot_id}` | {attempt['attempt']} | `{outcome}` | "
                f"`{attempt.get('status')}` | {video_markdown_link(root, video)} | `{errors}` |"
            )
    (root / "batch-index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconcile_interrupted_attempts(root: Path, state: Dict[str, Any]) -> int:
    reconciled = 0
    start_condition = state["start_condition"]
    slot_definitions = {slot["id"]: slot for slot in slots()}
    for slot_id, slot in slot_definitions.items():
        slot_state = state["slots"].setdefault(
            slot_id, {"slot": slot, "attempts": [], "accepted_attempt": None}
        )
        known_attempts = {attempt["attempt"] for attempt in slot_state.get("attempts", [])}
        slot_root = root / slot["symptom"] / slot["cause"] / slot["variant"]
        for attempt_dir in sorted(slot_root.glob("attempt-*")) if slot_root.is_dir() else []:
            try:
                attempt_number = int(attempt_dir.name.removeprefix("attempt-"))
            except ValueError:
                continue
            if attempt_number in known_attempts:
                continue
            record = inspect_attempt(
                root, slot, attempt_number, -15, [], start_condition
            )
            record["quality_passed"] = False
            record["status"] = "void"
            record.setdefault("quality_errors", []).append("attempt_interrupted_before_state_update")
            launcher_log = attempt_dir / "launcher.log"
            if launcher_log.is_file():
                record["launcher_log"] = {
                    "path": str(launcher_log),
                    "sha256": sha256(launcher_log),
                }
            slot_state["attempts"].append(record)
            reconciled += 1
        for attempt in slot_state.get("attempts", []):
            if attempt.get("status") == "running":
                attempt["status"] = "void"
                attempt["quality_passed"] = False
                attempt.setdefault("quality_errors", []).append("attempt_interrupted")
                reconciled += 1
    return reconciled


def main() -> int:
    args = parse_args()
    root = args.results_dir
    root.mkdir(parents=True, exist_ok=True)
    start_position = [float(value) for value in args.cube_initial_position]
    start_condition = {
        "id": args.start_condition_id,
        "cube_initial_position_m": start_position,
        "changed_from_default": {
            "cube_x_m": start_position[0] - 0.5,
            "cube_y_m": start_position[1],
            "cube_z_m": start_position[2] - 0.0258,
        },
    }
    try:
        state = load_state(root, args.resume, start_condition)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 2
    reconcile_interrupted_attempts(root, state)
    if args.reject_reason:
        state["condition_decision"] = {
            "accepted": False,
            "reason": args.reject_reason,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    if args.reconcile_only:
        write_reports(root, state)
        print(f"Reconciled interrupted attempts in {root / 'batch-state.json'}")
        return 0
    for slot in slots():
        slot_state = state["slots"].setdefault(slot["id"], {"slot": slot, "attempts": [], "accepted_attempt": None})
        if slot_state.get("accepted_attempt"):
            continue
        first_attempt = len(slot_state["attempts"]) + 1
        for attempt in range(first_attempt, args.max_attempts + 1):
            attempt_dir = root / slot["symptom"] / slot["cause"] / slot["variant"] / f"attempt-{attempt:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            command = command_for(args, slot, attempt_dir, state["start_condition"])
            launcher_log = attempt_dir / "launcher.log"
            running_record = {
                "attempt": attempt,
                "attempt_directory": str(attempt_dir),
                "command": command,
                "returncode": None,
                "session": None,
                "quality_passed": False,
                "quality_errors": [],
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            slot_state["attempts"].append(running_record)
            write_reports(root, state)
            with launcher_log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            record = inspect_attempt(
                root,
                slot,
                attempt,
                completed.returncode,
                command,
                state["start_condition"],
            )
            record["launcher_log"] = {"path": str(launcher_log), "sha256": sha256(launcher_log)}
            slot_state["attempts"][-1] = record
            if record["quality_passed"]:
                slot_state["accepted_attempt"] = record
                break
            record["status"] = "void"
            write_reports(root, state)
        write_reports(root, state)
        if not slot_state.get("accepted_attempt"):
            state["condition_decision"] = {
                "accepted": False,
                "reason": (
                    f"slot {slot['id']} failed its expected-outcome gate after "
                    f"{len(slot_state['attempts'])} attempts"
                ),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            write_reports(root, state)
            print(
                "Stopped because a version no longer met its expected-outcome gate; "
                "later slots were not generated."
            )
            return 1
    write_reports(root, state)
    complete = state["counts"]["accepted_records"] == state["counts"]["planned_slots"]
    state["execution_finished_at"] = datetime.now(timezone.utc).isoformat()
    unresolved_slots = [
        slot_id for slot_id, item in state["slots"].items()
        if not item.get("accepted_attempt")
    ]
    state["condition_decision"] = {
        "accepted": complete,
        "reason": (
            "all planned slots passed"
            if complete
            else f"outcome pattern did not hold for: {', '.join(unresolved_slots)}"
        ),
        "recorded_at": state["execution_finished_at"],
    }
    write_reports(root, state)
    print(f"Wrote bounded start-condition batch to {root / 'batch-index.md'}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
