"""Run the three-cause, three-treatment mid-transport drop experiment.

For each cause this launches exactly three matched runs:
``injected-fault`` (cause active), ``wrong-cause`` (an ineffective diagnosis
while the cause remains active), and ``true-fix`` (only the real cause removed).
Every child writes the same per-run artifacts as the existing experiments.
"""

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
    "gripper-release": {
        "label": "Gripper opens during transport",
        "wrong_diagnosis": "contact_friction",
        "true_cause": "gripper_command_open_during_transport",
    },
    "friction-loss": {
        "label": "Finger-cube collision is disabled during transport",
        "wrong_diagnosis": "gripper_command",
        "true_cause": "finger_cube_collision_disabled_during_transport",
    },
    "motion-disturbance": {
        "label": "Downward motion disturbance during transport",
        "wrong_diagnosis": "gripper_command",
        "true_cause": "downward_disturbance_velocity",
    },
}
VARIANTS = (
    ("injected-fault", "Fault active"),
    ("wrong-cause", "Wrong-cause handling"),
    ("true-fix", "True-cause fix"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("artifacts/midway-drop-comparison"))
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

    def rel(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(root.resolve()))
        except ValueError:
            return str(path)

    record.update(
        {
            "summary": rel(summary_path) if summary_path.is_file() else None,
            "result": rel(result_path) if result_path.is_file() else None,
            "log": rel(log_path) if log_path.is_file() else None,
            "trace": rel(trace_path) if trace_path.is_file() else None,
            "video": rel(video_path) if video_path.is_file() else None,
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
        "--midway-drop-cause",
        cause,
        "--midway-drop-variant",
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
    try:
        record["launcher_log"] = str(launcher_log.resolve().relative_to(root.resolve()))
    except ValueError:
        record["launcher_log"] = str(launcher_log)
    record.setdefault("sha256", {})["launcher_log"] = sha256(launcher_log)
    return record


def first_transport_phase(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    for phase in (result or {}).get("phase_history", []):
        if phase.get("phase") == "move_to_target":
            return phase
    return {}


def fairness(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("seed", "device", "ik_method", "reset_settle_steps", "post_run_settle_steps", "task_contract")
    checks = {}
    for key in keys:
        encoded = {
            json.dumps((record.get("config") or {}).get(key), sort_keys=True)
            for record in records.values()
        }
        checks[key] = {"equal": len(encoded) == 1, "values": [json.loads(value) for value in encoded]}
    return checks


def trace_prefix_checks(root: Path, records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def path(record: Dict[str, Any]) -> Optional[Path]:
        value = record.get("trace")
        return root / value if value and not Path(value).is_absolute() else Path(value) if value else None

    def read(record: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        trace = path(record)
        return [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()] if trace and trace.is_file() else None

    result: Dict[str, Any] = {}
    for cause in CAUSES:
        normal = read(records.get(cause + "/injected-fault", {}))
        wrong = read(records.get(cause + "/wrong-cause", {}))
        fixed = read(records.get(cause + "/true-fix", {}))
        prefix = 176
        keys = ("label", "run_step", "event", "phase", "phase_step", "end_effector_position", "end_effector_orientation", "dof_positions")
        def controls(items: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
            return [tuple(item.get(key) for key in keys) for item in items[:prefix]] if items is not None else None
        result[cause] = {
            "fault_wrong_control_prefix_equal": controls(normal) == controls(wrong),
            "fault_fixed_control_prefix_equal": controls(normal) == controls(fixed),
            "prefix_observation_count": prefix,
        }
    return result


def write_report(root: Path, records: Dict[str, Dict[str, Any]]) -> None:
    fairness_checks = fairness(records)
    all_equal = all(check["equal"] and check["values"] != [None] for check in fairness_checks.values())
    trace = trace_prefix_checks(root, records)
    manifest = {
        "schema": "midway-drop-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "symptom": "cube is grasped, then drops during transport",
        "causes": CAUSES,
        "fairness": fairness_checks,
        "trace_checks": trace,
        "variants": records,
    }
    causal_checks: Dict[str, Any] = {"all_task_parameters_equal": all_equal}
    lines = [
        "# Midway-drop three-cause comparison",
        "",
        "Every cause uses the same robot, cube, reset pose, seed, target, phase durations, IK, and settling steps.",
        "Each fault is triggered at the start of `move_to_target`, after the lift phase, so the cube must first be grasped.",
        "",
    ]
    for cause, info in CAUSES.items():
        lines.extend([f"## {info['label']} (`{cause}`)", "", "| Variant | Success | Grasped before fault | Drop detected | Pre-drop lift (m) | Drop distance (m) | EE separation increase (m) | Action | Failure reasons | Video | Run record | Log |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |"])
        cause_records = {variant: records.get(f"{cause}/{variant}", {}) for variant, _ in VARIANTS}
        for variant, label in VARIANTS:
            record = cause_records[variant]
            run = record.get("run") or {}
            metrics = run.get("metrics") or {}
            phase = first_transport_phase(run)
            reasons = ", ".join(metrics.get("failure_reasons", [])) or "none"
            pre_drop_lift = metrics.get("pre_drop_lift_m")
            drop_distance = metrics.get("drop_distance_m")
            separation_increase = metrics.get("drop_separation_increase_m")
            pre_drop_text = f"{pre_drop_lift:.6f}" if isinstance(pre_drop_lift, (int, float)) else "n/a"
            drop_distance_text = f"{drop_distance:.6f}" if isinstance(drop_distance, (int, float)) else "n/a"
            separation_text = f"{separation_increase:.6f}" if isinstance(separation_increase, (int, float)) else "n/a"
            lines.append(
                f"| {label} | `{metrics.get('success')}` | `{metrics.get('grasp_established_before_fault')}` | `{metrics.get('drop_detected')}` | "
                f"{pre_drop_text} | {drop_distance_text} | {separation_text} | "
                f"`{metrics.get('drop_action') or 'none'}` | `{reasons}` | `{record.get('video') or 'missing video'}` | `{record.get('result') or 'missing result'}` | `{record.get('log') or 'missing log'}` |"
            )
        fault_record = cause_records["injected-fault"]
        wrong_record = cause_records["wrong-cause"]
        fix_record = cause_records["true-fix"]
        fault_metrics = (fault_record.get("run") or {}).get("metrics") or {}
        wrong_metrics = (wrong_record.get("run") or {}).get("metrics") or {}
        fix_metrics = (fix_record.get("run") or {}).get("metrics") or {}
        fault_phase, wrong_phase, fix_phase = first_transport_phase(fault_record.get("run")), first_transport_phase(wrong_record.get("run")), first_transport_phase(fix_record.get("run"))
        cause_checks = {
            "fault_failed_after_grasp": bool(not fault_metrics.get("success") and fault_metrics.get("grasp_established_before_fault") and fault_metrics.get("drop_detected")),
            "wrong_cause_failed_with_cause_preserved": bool(not wrong_metrics.get("success") and wrong_metrics.get("drop_detected") and wrong_phase.get("true_cause_preserved") is True),
            "true_fix_succeeded_without_drop": bool(fix_metrics.get("success") and not fix_metrics.get("drop_detected") and fix_phase.get("true_cause_modified") is True),
            "control_prefixes_match": trace[cause]["fault_wrong_control_prefix_equal"] and trace[cause]["fault_fixed_control_prefix_equal"],
        }
        causal_checks[cause] = cause_checks
        if cause == "friction-loss":
            root_cause_evidence = (
                f"fault action=`{fault_metrics.get('drop_action')}`, "
                f"collision filter before/after=`{fault_metrics.get('finger_cube_collision_filter_before')}`/"
                f"`{fault_metrics.get('finger_cube_collision_filter_after')}`, "
                f"wrong diagnosis=`{wrong_phase.get('wrong_diagnosis')}`, "
                f"true cause preserved in wrong run=`{wrong_phase.get('true_cause_preserved')}`"
            )
        else:
            root_cause_evidence = (
                f"fault action=`{fault_metrics.get('drop_action')}`, "
                f"wrong diagnosis=`{wrong_phase.get('wrong_diagnosis')}`, "
                f"true cause preserved in wrong run=`{wrong_phase.get('true_cause_preserved')}`"
            )
        lines.extend([
            "",
            f"- Root-cause evidence: {root_cause_evidence}.",
            f"- Cause checks: fault drops after grasp **{cause_checks['fault_failed_after_grasp']}**, wrong-cause still drops **{cause_checks['wrong_cause_failed_with_cause_preserved']}**, true-fix recovers **{cause_checks['true_fix_succeeded_without_drop']}**.",
            f"- Control prefixes match before the intervention: **{cause_checks['control_prefixes_match']}**.",
            "",
        ])
    manifest["causal_checks"] = causal_checks
    (root / "comparison.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines.extend([
        f"- Shared task contract identical across all nine runs: **{all_equal}**.",
        f"- Complete video/log/result/trace evidence in all nine runs: **{all(record.get('evidence_complete') for record in records.values())}**.",
        f"- All experiment logs are free of ERROR/TRACEBACK/FATAL/Exception markers: **{all(record.get('experiment_log_clean') for record in records.values())}**.",
        "- MP4 and record SHA-256 values are in `comparison.json`.",
    ])
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
    print(f"Wrote midway-drop comparison to {args.results_dir / 'comparison.md'}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
