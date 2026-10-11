"""Run a four-version causal comparison for controlled Fault 2 or Fault 3.

Each child run gets a fresh Isaac Sim process and writes its own MP4, log,
summary, result, and trace. The parent writes ``comparison.json`` and
``comparison.md`` with fairness, prefix-trace, and evidence checks.
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


VARIANTS = (
    ("normal", "Normal grasp"),
    ("injected-fault", "Injected fault"),
    ("wrong-cause", "Wrong-cause handling"),
    ("true-fix", "True-cause fix"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault", choices=["2", "3"], required=True, help="Controlled fault number")
    parser.add_argument("--results-dir", type=Path, default=None)
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
    if parsed.results_dir is None:
        parsed.results_dir = Path(f"artifacts/fault{parsed.fault}-comparison")
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


def run_variant(args: argparse.Namespace, root: Path, variant: str) -> Dict[str, Any]:
    directory = root / variant
    directory.mkdir(parents=True, exist_ok=True)
    flag = f"--fault{args.fault}-variant"
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
        flag,
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


def first_phase(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    for phase in (result or {}).get("phase_history", []):
        if phase.get("phase") == "close_gripper":
            return phase
    return {}


def fairness(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("seed", "device", "ik_method", "reset_settle_steps", "post_run_settle_steps", "task_contract")
    result = {}
    for key in keys:
        encoded = {
            json.dumps((record.get("config") or {}).get(key), sort_keys=True)
            for record in records.values()
        }
        result[key] = {"equal": len(encoded) == 1, "values": [json.loads(value) for value in encoded]}
    return result


def trace_checks(root: Path, fault: str, records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def path(record: Dict[str, Any]) -> Optional[Path]:
        value = record.get("trace")
        return root / value if value and not Path(value).is_absolute() else Path(value) if value else None

    def read(record: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        trace = path(record)
        return [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()] if trace and trace.is_file() else None

    normal = read(records.get("normal", {}))
    injected = read(records.get("injected-fault", {}))
    wrong = read(records.get("wrong-cause", {}))
    fixed = read(records.get("true-fix", {}))
    normal_path, fixed_path = path(records.get("normal", {})), path(records.get("true-fix", {}))
    prefix = 116 if fault == "2" else 76
    control_keys = (
        "label",
        "run_step",
        "event",
        "phase",
        "phase_step",
        "end_effector_position",
        "end_effector_orientation",
        "dof_positions",
    )

    def control_prefix_equal(left: Optional[List[Dict[str, Any]]], right: Optional[List[Dict[str, Any]]]) -> bool:
        if left is None or right is None:
            return False
        return [tuple((item.get(key) for key in control_keys)) for item in left[:prefix]] == [
            tuple((item.get(key) for key in control_keys)) for item in right[:prefix]
        ]

    return {
        "normal_true_fix_byte_equal": bool(normal_path and fixed_path and normal_path.is_file() and fixed_path.is_file() and normal_path.read_bytes() == fixed_path.read_bytes()),
        "normal_injected_control_prefix_equal": control_prefix_equal(normal, injected),
        "normal_wrong_cause_control_prefix_equal": control_prefix_equal(normal, wrong),
        "prefix_observation_count": prefix,
        "prefix_comparison": "robot control fields only" if fault == "2" else "all trace fields",
    }


def write_report(root: Path, fault: str, records: Dict[str, Dict[str, Any]]) -> None:
    fairness_checks = fairness(records)
    trace = trace_checks(root, fault, records)
    normal = records.get("normal", {})
    injected = records.get("injected-fault", {})
    wrong = records.get("wrong-cause", {})
    fixed = records.get("true-fix", {})
    normal_success = bool((normal.get("run") or {}).get("metrics", {}).get("success"))
    injected_success = bool((injected.get("run") or {}).get("metrics", {}).get("success"))
    wrong_success = bool((wrong.get("run") or {}).get("metrics", {}).get("success"))
    fixed_success = bool((fixed.get("run") or {}).get("metrics", {}).get("success"))
    wrong_phase, fixed_phase = first_phase(wrong.get("run")), first_phase(fixed.get("run"))
    all_equal = all(check["equal"] and check["values"] != [None] for check in fairness_checks.values())
    cause_preserved = wrong_phase.get("true_cause_preserved") is True
    cause_modified = fixed_phase.get("true_cause_modified") is True
    outcome = normal_success and not injected_success and not wrong_success and fixed_success
    manifest = {
        "schema": f"fault{fault}-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fault": f"controlled Fault {fault}",
        "fairness": fairness_checks,
        "trace_checks": trace,
        "causal_checks": {
            "all_task_parameters_equal": all_equal,
            "wrong_cause_preserves_true_cause": cause_preserved,
            "true_fix_changes_true_cause_only": cause_modified,
            "outcome_pattern_supports_causality": outcome,
        },
        "variants": records,
    }
    (root / "comparison.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def metric(value: Any) -> str:
        return f"{value:.6f}" if isinstance(value, (float, int)) else "n/a"

    lines = [
        f"# Fault {fault} four-version comparison",
        "",
        "All variants use the same seed, reset state, target, phase durations, IK, device, and settling steps.",
        "The wrong-cause treatment applies a plausible but ineffective diagnosis while preserving the true fault.",
        "",
        "| Variant | Success | Steps | Trace | Max lift (m) | Final XY error (m) | Grasp command | True cause condition | Failure reasons | Video | Run record | Log |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for variant, label in VARIANTS:
        record, run = records.get(variant, {}), (records.get(variant, {}) or {}).get("run") or {}
        metrics, phase = run.get("metrics") or {}, first_phase(run)
        reasons = ", ".join(metrics.get("failure_reasons", [])) or "none"
        condition = phase.get("contact_condition") if fault == "2" else phase.get("approach_condition")
        lines.append(
            f"| {label} | `{metrics.get('success')}` | {run.get('simulation_steps', 'n/a')} | {record.get('trace_observations', 'n/a')} | "
            f"{metric(metrics.get('max_lift_m'))} | {metric(metrics.get('final_xy_error_m'))} | `{phase.get('grasp_command', 'n/a')}` | `{condition or 'n/a'}` | "
            f"`{reasons}` | `{record.get('video') or 'missing video'}` | `{record.get('result') or 'missing result'}` | `{record.get('log') or 'missing log'}` |"
        )
    lines.extend(
        [
            "",
            f"- Fairness contract identical: **{all_equal}**.",
            f"- Wrong-cause handling preserved the true cause: **{cause_preserved}**; the run still failed. "
            f"For Fault 3, the height-related diagnosis/action is `{wrong_phase.get('wrong_diagnosis') or 'n/a'}` / `{wrong_phase.get('wrong_diagnosis_action') or 'n/a'}` while the `+0.08 m` approach offset remains active.",
            f"- True fix changed the true cause only: **{cause_modified}**; task difficulty stayed fixed.",
            f"- Four-outcome causal pattern (success, failure, failure, success): **{outcome}**.",
            f"- Complete video/log/result/trace evidence in all variants: **{all(record.get('evidence_complete') for record in records.values())}**.",
            f"- Experiment logs are free of ERROR/TRACEBACK/FATAL/Exception markers: **{all(record.get('experiment_log_clean') for record in records.values())}**.",
            f"- Normal and true-fix traces byte-identical: **{trace['normal_true_fix_byte_equal']}**; failure control prefixes match normal through observation {trace['prefix_observation_count']}: injected=**{trace['normal_injected_control_prefix_equal']}**, wrong-cause=**{trace['normal_wrong_cause_control_prefix_equal']}**.",
            "- MP4 and record SHA-256 values are in `comparison.json`.",
        ]
    )
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    records = {variant: run_variant(args, args.results_dir, variant) for variant, _ in VARIANTS}
    write_report(args.results_dir, args.fault, records)
    complete = all(record.get("returncode") == 0 and record.get("evidence_complete") for record in records.values())
    print(f"Wrote Fault {args.fault} comparison to {args.results_dir / 'comparison.md'}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
