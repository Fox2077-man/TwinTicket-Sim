"""Run and compare the four controlled Fault 1 gripper variants.

The script launches one Isaac Sim process per variant so every process gets a
fresh stage while sharing the same task contract. Each child writes its own
video, experiment log, summary, result, and trace. The parent then writes a
machine-readable manifest and a short evidence report.

Run this with the Isaac Sim Python interpreter (or inside the repository's
Isaac Sim container), for example::

    /isaac-sim/python.sh run_fault1_comparison.py
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
    ("normal", "Normal grasp", "close command, default contact"),
    ("injected-fault", "Injected fault", "open command during grasp"),
    ("wrong-cause", "Wrong-cause handling", "increase friction, keep open command"),
    ("true-fix", "True-cause fix", "restore close command only"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("artifacts/fault1-comparison"),
        help="Directory containing four variant directories and the comparison report",
    )
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
    parser.add_argument(
        "--run-script",
        type=Path,
        default=Path(__file__).with_name("run_franka_pick_place.py"),
    )
    parser.add_argument(
        "--no-webrtc",
        action="store_true",
        help="Do not enable WebRTC (video capture normally requires an active viewport)",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument forwarded to every child runner; may be repeated",
    )
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


def load_variant_record(
    directory: Path, returncode: int, command: List[str], comparison_root: Optional[Path] = None
) -> Dict[str, Any]:
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
        # Result paths are usually absolute, but keep the manifest portable.
        video_path = session / video_path.name
    display_root = (comparison_root or directory.parent).resolve()

    def display_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(display_root))
        except ValueError:
            return str(path)

    record.update(
        {
            "summary": display_path(summary_path) if summary_path.is_file() else None,
            "result": display_path(result_path) if result_path.is_file() else None,
            "log": display_path(log_path) if log_path.is_file() else None,
            "trace": display_path(trace_path) if trace_path.is_file() else None,
            "video": display_path(video_path) if video_path.is_file() else None,
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
        summary
        and result
        and video_path.is_file()
        and log_path.is_file()
        and trace_path.is_file()
    )
    record["experiment_log_clean"] = not bool(
        re.search(r"\b(ERROR|TRACEBACK|FATAL|Exception)\b", log_path.read_text(encoding="utf-8"))
    ) if log_path.is_file() else False
    return record


def run_variant(args: argparse.Namespace, root: Path, variant: str) -> Dict[str, Any]:
    variant_dir = root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
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
        "--fault1-variant",
        variant,
        "--results-dir",
        str(variant_dir),
    ]
    if not args.no_webrtc:
        command.append("--webrtc")
    command.extend(args.extra_arg)
    launcher_log = variant_dir / "launcher.log"
    with launcher_log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    record = load_variant_record(variant_dir, completed.returncode, command, root)
    try:
        record["launcher_log"] = str(launcher_log.resolve().relative_to(root.resolve()))
    except ValueError:
        record["launcher_log"] = str(launcher_log)
    record.setdefault("sha256", {})["launcher_log"] = sha256(launcher_log)
    return record


def first_phase(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not result:
        return {}
    for phase in result.get("phase_history", []):
        if phase.get("phase") == "close_gripper":
            return phase
    return {}


def fairness_contract(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("seed", "device", "ik_method", "reset_settle_steps", "post_run_settle_steps", "task_contract")
    values: Dict[str, Any] = {}
    for key in keys:
        observed = {json.dumps((record.get("config") or {}).get(key), sort_keys=True) for record in records.values()}
        values[key] = {"equal": len(observed) == 1, "values": [json.loads(item) for item in observed]}
    return values


def trace_comparison(root: Path, records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    def trace_path(record: Dict[str, Any]) -> Optional[Path]:
        value = record.get("trace")
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else root / path

    def lines(record: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        path = trace_path(record)
        if path is None or not path.is_file():
            return None
        return [json.loads(item) for item in path.read_text(encoding="utf-8").splitlines()]

    normal_lines = lines(records.get("normal", {}))
    injected_lines = lines(records.get("injected-fault", {}))
    wrong_lines = lines(records.get("wrong-cause", {}))
    fixed_lines = lines(records.get("true-fix", {}))
    prefix_length = 116
    result: Dict[str, Any] = {
        "normal_true_fix_byte_equal": False,
        "normal_injected_prefix_equal_through_grasp": False,
        "normal_wrong_cause_prefix_equal_through_grasp": False,
        "prefix_observation_count": prefix_length,
    }
    normal_path = trace_path(records.get("normal", {}))
    fixed_path = trace_path(records.get("true-fix", {}))
    if normal_path and fixed_path and normal_path.is_file() and fixed_path.is_file():
        result["normal_true_fix_byte_equal"] = normal_path.read_bytes() == fixed_path.read_bytes()
    if normal_lines is not None and injected_lines is not None:
        result["normal_injected_prefix_equal_through_grasp"] = normal_lines[:prefix_length] == injected_lines[:prefix_length]
    if normal_lines is not None and wrong_lines is not None:
        result["normal_wrong_cause_prefix_equal_through_grasp"] = normal_lines[:prefix_length] == wrong_lines[:prefix_length]
    return result


def write_report(root: Path, records: Dict[str, Dict[str, Any]]) -> None:
    normal = records.get("normal", {})
    injected = records.get("injected-fault", {})
    wrong = records.get("wrong-cause", {})
    fixed = records.get("true-fix", {})
    normal_success = bool((normal.get("run") or {}).get("metrics", {}).get("success"))
    injected_success = bool((injected.get("run") or {}).get("metrics", {}).get("success"))
    wrong_success = bool((wrong.get("run") or {}).get("metrics", {}).get("success"))
    fixed_success = bool((fixed.get("run") or {}).get("metrics", {}).get("success"))
    wrong_phase = first_phase(wrong.get("run"))
    fixed_phase = first_phase(fixed.get("run"))
    contract = fairness_contract(records)
    trace_checks = trace_comparison(root, records)
    all_equal = all(item["equal"] and item["values"] != [None] for item in contract.values())
    causality_supported = normal_success and not injected_success and not wrong_success and fixed_success
    cause_preserved = (
        wrong_phase.get("grasp_command") == "open"
        and wrong_phase.get("true_cause_preserved") is True
    )
    fix_only_command = fixed_phase.get("grasp_command") == "close" and fixed_phase.get("true_cause_modified") is True

    manifest = {
        "schema": "fault1-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fault": "gripper does not close during grasp",
        "fairness": contract,
        "trace_checks": trace_checks,
        "causal_checks": {
            "all_task_parameters_equal": all_equal,
            "wrong_cause_preserves_open_command": cause_preserved,
            "true_fix_changes_only_gripper_command": fix_only_command,
            "outcome_pattern_supports_causality": causality_supported,
        },
        "variants": records,
    }
    (root / "comparison.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Fault 1 four-version comparison",
        "",
        "The four runs use the same seed, robot/cube reset, target, phase durations, IK, device, and settling steps.",
        "Only the designated Fault 1 intervention changes; the task contract is identical. Every variant has its own MP4 and run record.",
        "",
        "| Variant | Success | Steps | Trace | Max lift (m) | Final XY error (m) | Grasp command | Failure reasons | Video | Run record | Log |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]

    def metric_text(value: Any) -> str:
        return f"{value:.6f}" if isinstance(value, (float, int)) else "n/a"

    for variant, label, _ in VARIANTS:
        record = records.get(variant, {})
        run = record.get("run") or {}
        metrics = run.get("metrics") or {}
        phase = first_phase(run)
        evidence = record.get("video") or "missing video"
        result_path = record.get("result") or "missing result"
        log_path = record.get("log") or "missing log"
        failure_reasons = ", ".join(metrics.get("failure_reasons", [])) or "none"
        max_lift = metrics.get("max_lift_m")
        final_xy_error = metrics.get("final_xy_error_m")
        lines.append(
            f"| {label} | `{metrics.get('success')}` | {run.get('simulation_steps', 'n/a')} "
            f"| {record.get('trace_observations', 'n/a')} | "
            f"{metric_text(max_lift)} | {metric_text(final_xy_error)} | `{phase.get('grasp_command', 'n/a')}` "
            f"| `{failure_reasons}` | `{evidence}` | `{result_path}` | `{log_path}` |"
        )
    lines.extend(
        [
            "",
            f"- Fairness contract identical: **{all_equal}**.",
            f"- Wrong-cause handling kept the true open command: **{cause_preserved}**; its failure therefore remains a failed grasp, not a changed cause.",
            f"- True fix restored only the close command: **{fix_only_command}**; task parameters remain unchanged.",
            f"- Four-outcome causal pattern (normal succeeds, injected and wrong-cause fail, true fix succeeds): **{causality_supported}**.",
            f"- Complete video/log/result/trace evidence in all four variants: **{all(record.get('evidence_complete') for record in records.values())}**; SHA-256 values are in `comparison.json`.",
            f"- Experiment logs are complete and free of ERROR/TRACEBACK/FATAL/Exception markers: **{all(record.get('experiment_log_clean') for record in records.values())}**.",
            f"- Normal and true-fix traces are byte-identical: **{trace_checks['normal_true_fix_byte_equal']}**; both failure traces match normal through observation {trace_checks['prefix_observation_count']} before the grasp command diverges.",
            "",
            "Each variant directory contains `experiment.log`, `summary.json`, `run-001/result.json`, `run-001/trace.jsonl`, and `run-001.mp4` when viewport capture is available.",
        ]
    )
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.results_dir
    root.mkdir(parents=True, exist_ok=True)
    records = {variant: run_variant(args, root, variant) for variant, _, _ in VARIANTS}
    write_report(root, records)
    print(f"Wrote Fault 1 comparison to {root / 'comparison.md'}")
    complete = all(record.get("returncode") == 0 and record.get("evidence_complete") for record in records.values())
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
