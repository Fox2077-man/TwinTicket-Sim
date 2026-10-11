"""Run the three-cause inaccurate-placement comparison with matched evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


CAUSES = {
    "planner-x-offset": {
        "label": "Planner release target has an X offset",
        "true_cause": "planner_release_target_x_offset",
        "wrong_diagnosis": "gripper_closure",
    },
    "transport-y-bias": {
        "label": "Transport controller command has a Y bias",
        "true_cause": "transport_controller_y_command_bias",
        "wrong_diagnosis": "planner_target_frame",
    },
    "early-release": {
        "label": "Gripper opens before reaching the target",
        "true_cause": "open_gripper_before_target",
        "wrong_diagnosis": "transport_tracking",
    },
}
VARIANTS = (
    ("injected-fault", "Fault active"),
    ("wrong-cause", "Wrong-cause handling"),
    ("true-fix", "True-cause fix"),
)
TRACE_PREFIX_OBSERVATIONS = 176


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/misplacement-comparison"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--ik-method",
        choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
        default="damped-least-squares",
    )
    parser.add_argument("--reset-settle-steps", type=int, default=15)
    parser.add_argument("--post-run-settle-steps", type=int, default=30)
    parser.add_argument("--sim-python", default=sys.executable, help="Isaac Sim Python interpreter")
    parser.add_argument("--run-script", type=Path, default=Path(__file__).with_name("run_franka_pick_place.py"))
    parser.add_argument("--no-webrtc", action="store_true", help="Do not enable WebRTC")
    parser.add_argument("--extra-arg", action="append", default=[], help="Argument forwarded to every child")
    parsed = parser.parse_args()
    if parsed.reset_settle_steps < 0 or parsed.post_run_settle_steps < 0:
        parser.error("settle step counts must be non-negative")
    return parsed


def sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_session(directory: Path) -> Optional[Path]:
    sessions = [item for item in directory.glob("session-*") if item.is_dir()]
    return max(sessions, key=lambda item: item.stat().st_mtime) if sessions else None


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_record(directory: Path, root: Path, returncode: int, command: List[str]) -> Dict[str, Any]:
    session = latest_session(directory)
    record: Dict[str, Any] = {
        "directory": str(directory),
        "returncode": returncode,
        "command": command,
        "session": str(session) if session else None,
    }
    if session is None:
        return record

    summary_path = session / "summary.json"
    result_path = session / "run-001" / "result.json"
    log_path = session / "experiment.log"
    trace_path = session / "run-001" / "trace.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
    video_path = Path(result["video"]) if result and result.get("video") else session / "run-001.mp4"
    if not video_path.is_file():
        video_path = session / video_path.name

    record.update(
        {
            "summary": relative_to_root(summary_path, root) if summary_path.is_file() else None,
            "result": relative_to_root(result_path, root) if result_path.is_file() else None,
            "log": relative_to_root(log_path, root) if log_path.is_file() else None,
            "trace": relative_to_root(trace_path, root) if trace_path.is_file() else None,
            "video": relative_to_root(video_path, root) if video_path.is_file() else None,
            "sha256": {
                "video": sha256(video_path),
                "log": sha256(log_path),
                "result": sha256(result_path),
                "trace": sha256(trace_path),
            },
            "config": summary.get("config") if summary else None,
            "run": result,
            "trace_observations": len(trace_path.read_text(encoding="utf-8").splitlines())
            if trace_path.is_file()
            else None,
        }
    )
    record["evidence_complete"] = bool(
        summary and result and video_path.is_file() and log_path.is_file() and trace_path.is_file()
    )
    record["experiment_log_clean"] = (
        not bool(re.search(r"\b(ERROR|TRACEBACK|FATAL|Exception)\b", log_path.read_text(encoding="utf-8")))
        if log_path.is_file()
        else False
    )
    return record


def run_variant(args: argparse.Namespace, root: Path, cause: str, variant: str) -> Dict[str, Any]:
    directory = root / cause / variant
    directory.mkdir(parents=True, exist_ok=True)
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
        "--misplacement-cause",
        cause,
        "--misplacement-variant",
        variant,
        "--results-dir",
        str(directory),
    ]
    if not args.no_webrtc:
        command.append("--webrtc")
    command.extend(args.extra_arg)
    launcher_log = directory / "launcher.log"
    with launcher_log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    record = load_record(directory, root, completed.returncode, command)
    record["launcher_log"] = relative_to_root(launcher_log, root)
    record.setdefault("sha256", {})["launcher_log"] = sha256(launcher_log)
    return record


def placement_phase(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    for phase in (result or {}).get("phase_history", []):
        if phase.get("phase") == "move_to_target":
            return phase
    return {}


def fairness(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("seed", "device", "ik_method", "reset_settle_steps", "post_run_settle_steps", "task_contract")
    checks: Dict[str, Any] = {}
    for key in keys:
        encoded = {
            json.dumps((record.get("config") or {}).get(key), sort_keys=True)
            for record in records.values()
        }
        checks[key] = {"equal": len(encoded) == 1, "values": [json.loads(value) for value in encoded]}
    return checks


def trace_prefix_checks(root: Path, records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def trace(record: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        location = record.get("trace")
        path = root / location if location and not Path(location).is_absolute() else Path(location) if location else None
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path and path.is_file() else None

    keys = ("label", "run_step", "event", "phase", "phase_step", "end_effector_position", "end_effector_orientation", "dof_positions")

    def controls(items: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
        return [tuple(item.get(key) for key in keys) for item in items[:TRACE_PREFIX_OBSERVATIONS]] if items else None

    checks = {}
    for cause in CAUSES:
        injected = controls(trace(records.get(f"{cause}/injected-fault", {})))
        wrong = controls(trace(records.get(f"{cause}/wrong-cause", {})))
        fixed = controls(trace(records.get(f"{cause}/true-fix", {})))
        checks[cause] = {
            "fault_wrong_control_prefix_equal": injected == wrong,
            "fault_fixed_control_prefix_equal": injected == fixed,
            "prefix_observation_count": TRACE_PREFIX_OBSERVATIONS,
        }
    return checks


def vector(value: Any) -> str:
    return json.dumps(value) if isinstance(value, list) else "n/a"


def write_report(root: Path, records: Dict[str, Dict[str, Any]]) -> None:
    fairness_checks = fairness(records)
    all_equal = all(check["equal"] and check["values"] != [None] for check in fairness_checks.values())
    trace_checks = trace_prefix_checks(root, records)
    manifest: Dict[str, Any] = {
        "schema": "misplacement-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symptom": "cube is grasped and transported, but is released away from the intended target",
        "causes": CAUSES,
        "fairness": fairness_checks,
        "trace_checks": trace_checks,
        "variants": records,
    }
    causal_checks: Dict[str, Any] = {"all_task_parameters_equal": all_equal}
    lines = [
        "# Inaccurate-placement three-cause comparison",
        "",
        "Every cause uses the same robot, cube, reset pose, seed, nominal target, phase durations, IK, and settling steps.",
        "The cube is grasped and lifted before the transport-phase intervention. The true-fix removes only the selected cause.",
        "",
    ]
    for cause, info in CAUSES.items():
        lines.extend(
            [
                f"## {info['label']} (`{cause}`)",
                "",
                "| Variant | Success | Lifted | Drop detected | Final XY error (m) | Action | Planned release target | Transport command | Video | Run record | Log |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
            ]
        )
        cause_records = {variant: records.get(f"{cause}/{variant}", {}) for variant, _ in VARIANTS}
        for variant, label in VARIANTS:
            record = cause_records[variant]
            metrics = (record.get("run") or {}).get("metrics") or {}
            error = metrics.get("final_xy_error_m")
            error_text = f"{error:.6f}" if isinstance(error, (int, float)) else "n/a"
            lines.append(
                f"| {label} | `{metrics.get('success')}` | `{metrics.get('lifted')}` | `{metrics.get('drop_detected')}` | "
                f"{error_text} | `{metrics.get('placement_action') or 'none'}` | "
                f"`{vector(metrics.get('planned_release_position_m'))}` | "
                f"`{vector(metrics.get('commanded_transport_position_m'))}` | "
                f"`{record.get('video') or 'missing video'}` | `{record.get('result') or 'missing result'}` | "
                f"`{record.get('log') or 'missing log'}` |"
            )

        fault = cause_records["injected-fault"]
        wrong = cause_records["wrong-cause"]
        fixed = cause_records["true-fix"]
        fault_metrics = (fault.get("run") or {}).get("metrics") or {}
        wrong_metrics = (wrong.get("run") or {}).get("metrics") or {}
        fixed_metrics = (fixed.get("run") or {}).get("metrics") or {}
        fault_phase = placement_phase(fault.get("run"))
        wrong_phase = placement_phase(wrong.get("run"))
        fixed_phase = placement_phase(fixed.get("run"))
        cause_check = {
            "fault_misplaced_after_grasp_and_transport": bool(
                not fault_metrics.get("success")
                and fault_metrics.get("lifted")
                and not fault_metrics.get("drop_detected")
                and fault_metrics.get("misplacement_detected")
            ),
            "wrong_cause_still_misplaced_with_cause_preserved": bool(
                not wrong_metrics.get("success")
                and wrong_metrics.get("misplacement_detected")
                and wrong_phase.get("misplacement_true_cause_preserved") is True
            ),
            "true_fix_succeeds_at_target": bool(
                fixed_metrics.get("success")
                and not fixed_metrics.get("misplacement_detected")
                and fixed_phase.get("misplacement_true_cause_modified") is True
            ),
            "control_prefixes_match": trace_checks[cause]["fault_wrong_control_prefix_equal"]
            and trace_checks[cause]["fault_fixed_control_prefix_equal"],
        }
        causal_checks[cause] = cause_check
        if cause == "early-release":
            evidence = (
                f"fault action=`{fault_metrics.get('placement_action')}`, "
                f"early release phase step=`{fault_metrics.get('early_release_phase_step')}`, "
                f"actual trigger=`{fault_metrics.get('early_release_triggered')}`, "
                f"trigger distance to target XY (m)=`{fault_metrics.get('early_release_distance_to_target_xy_m')}`"
            )
        else:
            evidence = (
                f"fault action=`{fault_metrics.get('placement_action')}`, "
                f"planner offset=`{fault_metrics.get('planner_offset_m')}`, "
                f"transport command bias=`{fault_metrics.get('transport_command_bias_m')}`"
            )
        lines.extend(
            [
                "",
                f"- Root-cause evidence: {evidence}.",
                f"- Wrong diagnosis/action: `{wrong_phase.get('misplacement_wrong_diagnosis')}`/"
                f"`{wrong_phase.get('misplacement_wrong_diagnosis_action')}`; true cause preserved="
                f"`{wrong_phase.get('misplacement_true_cause_preserved')}`.",
                f"- Cause checks: fault misplaces after grasp **{cause_check['fault_misplaced_after_grasp_and_transport']}**, "
                f"wrong-cause remains misplaced **{cause_check['wrong_cause_still_misplaced_with_cause_preserved']}**, "
                f"true-fix recovers **{cause_check['true_fix_succeeds_at_target']}**.",
                f"- Control prefixes match before the intervention: **{cause_check['control_prefixes_match']}**.",
                "",
            ]
        )

    manifest["causal_checks"] = causal_checks
    (root / "comparison.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines.extend(
        [
            f"- Shared task contract identical across all nine runs: **{all_equal}**.",
            f"- Complete video/log/result/trace evidence in all nine runs: **{all(record.get('evidence_complete') for record in records.values())}**.",
            f"- All experiment logs are free of ERROR/TRACEBACK/FATAL/Exception markers: **{all(record.get('experiment_log_clean') for record in records.values())}**.",
            "- MP4 and record SHA-256 values are in `comparison.json`.",
        ]
    )
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    records: Dict[str, Dict[str, Any]] = {}
    for cause in CAUSES:
        for variant, _ in VARIANTS:
            records[f"{cause}/{variant}"] = run_variant(args, args.results_dir, cause, variant)
    write_report(args.results_dir, records)
    complete = all(record.get("returncode") == 0 and record.get("evidence_complete") for record in records.values())
    print(f"Wrote inaccurate-placement comparison to {args.results_dir / 'comparison.md'}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
