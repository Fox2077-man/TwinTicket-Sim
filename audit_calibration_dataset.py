"""Freeze and audit the canonical 30-session calibration dataset.

The command has two modes:

* ``--initialize`` validates the current selected sessions and writes an
  immutable-by-default reference manifest with SHA-256 checksums.
* ``--verify`` rechecks the same 30 source sessions against that reference.

It never runs Isaac Sim and never copies or changes an experiment artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


REQUIRED_FILES = {
    "video": "run-001.mp4",
    "run_record": "run-001/result.json",
    "trace": "run-001/trace.jsonl",
    "experiment_log": "experiment.log",
    "session_summary": "summary.json",
}
VARIANTS = ("normal", "injected-fault", "wrong-cause", "true-fix")
CONTAINER_WORKSPACE = "/workspace/TwinTicket-Sim/"
TIMING_TOLERANCE_SECONDS = 10.0

SYMPTOMS: Mapping[str, Dict[str, Any]] = {
    "cannot-grasp": {
        "label": "抓不到",
        "causes": {
            "gripper-command-open": ("抓夹未关闭", "fault1_variant"),
            "zero-cube-friction": ("方块接触摩擦为零", "fault2_variant"),
            "high-grasp-approach": ("抓取高度偏高", "fault3_variant"),
        },
        "failure_visual": "抓取阶段未建立有效抓取，方块未被带起。",
        "success_visual": "方块被抓取并放入目标容差内。",
    },
    "midway-drop": {
        "label": "中途掉落",
        "causes": {
            "gripper-release": ("运输中打开抓夹", "midway_drop"),
            "friction-loss": ("运输中禁用指块碰撞", "midway_drop"),
            "motion-disturbance": ("运输中向下速度扰动", "midway_drop"),
        },
        "failure_visual": "方块先被抓住并抬起，随后在运输阶段脱离夹爪。",
        "success_visual": "方块在运输阶段保持被抓住，并放入目标容差内。",
    },
    "inaccurate-placement": {
        "label": "抓住并移动，但放置不准",
        "causes": {
            "planner-x-offset": ("规划释放目标 X 偏移", "misplacement"),
            "transport-y-bias": ("运输控制命令 Y 偏差", "misplacement"),
            "early-release": ("到目标前打开抓夹", "misplacement"),
        },
        "failure_visual": "方块完成抓取和运输，但最终落点在目标容差外。",
        "success_visual": "方块完成抓取、运输并落入目标容差内。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration-30"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--initialize", action="store_true", help="Create the reference manifest; refuse to overwrite it")
    mode.add_argument("--verify", action="store_true", help="Verify source files against the frozen reference manifest")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def resolve_repo_path(root: Path, value: Path | str) -> Path:
    """Resolve evidence paths written either on the host or in the container."""
    path = Path(value)
    if path.is_absolute() and str(path).startswith(CONTAINER_WORKSPACE):
        return root / str(path).removeprefix(CONTAINER_WORKSPACE)
    return path if path.is_absolute() else root / path


def output_link(path: str, root: Path, output: Path) -> str:
    return str(Path(path).resolve().relative_to(output.resolve().parent)) if Path(path).is_absolute() else str(
        Path("..") / Path(path).relative_to("artifacts")
    )


def parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def time_text(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def json_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None, None if isinstance(value, dict) else "root_is_not_object"
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}:{exc}"


def mp4_duration_seconds(path: Path) -> Optional[float]:
    """Read the ISO BMFF movie-header duration without external media tools."""
    try:
        data = path.read_bytes()
    except OSError:
        return None

    def boxes(start: int, end: int) -> Iterable[Tuple[bytes, int, int]]:
        offset = start
        while offset + 8 <= end:
            size = struct.unpack_from(">I", data, offset)[0]
            kind = data[offset + 4 : offset + 8]
            header = 8
            if size == 1:
                if offset + 16 > end:
                    return
                size = struct.unpack_from(">Q", data, offset + 8)[0]
                header = 16
            elif size == 0:
                size = end - offset
            if size < header or offset + size > end:
                return
            yield kind, offset + header, offset + size
            offset += size

    for kind, payload_start, payload_end in boxes(0, len(data)):
        if kind != b"moov":
            continue
        for child_kind, child_start, child_end in boxes(payload_start, payload_end):
            if child_kind != b"mvhd" or child_start + 20 > child_end:
                continue
            version = data[child_start]
            if version == 0 and child_start + 20 <= child_end:
                timescale = struct.unpack_from(">I", data, child_start + 12)[0]
                duration = struct.unpack_from(">I", data, child_start + 16)[0]
            elif version == 1 and child_start + 32 <= child_end:
                timescale = struct.unpack_from(">I", data, child_start + 20)[0]
                duration = struct.unpack_from(">Q", data, child_start + 24)[0]
            else:
                return None
            return float(duration) / timescale if timescale else None
    return None


def canonical_ids() -> List[str]:
    expected: List[str] = []
    for symptom, spec in SYMPTOMS.items():
        expected.append(f"{symptom}/shared-normal/normal")
        for cause in spec["causes"]:
            expected.extend(f"{symptom}/{cause}/{variant}" for variant in VARIANTS[1:])
    return expected


def labels_for(symptom: str, cause: str, variant: str) -> Dict[str, Any]:
    spec = SYMPTOMS[symptom]
    normal = variant == "normal"
    success = normal or variant == "true-fix"
    cause_label = "共享正常抓取" if normal else spec["causes"][cause][0]
    variant_labels = {
        "normal": "正常抓取",
        "injected-fault": "注入故障",
        "wrong-cause": "错误归因失败",
        "true-fix": "正确归因成功",
    }
    return {
        "symptom": symptom,
        "symptom_label": spec["label"],
        "cause": cause,
        "cause_label": cause_label,
        "variant": variant,
        "variant_label": variant_labels[variant],
        "expected_success": success,
        "expected_video_observation": spec["success_visual"] if success else spec["failure_visual"],
    }


def load_selection(root: Path, index_path: Path) -> Tuple[List[Dict[str, Any]], List[str], str]:
    index, error = json_file(index_path)
    if error or index is None:
        return [], [f"selection_index_unreadable:{error}"], ""
    issues: List[str] = []
    sessions = index.get("sessions")
    if not isinstance(sessions, list):
        return [], ["selection_index_sessions_missing"], sha256(index_path)
    selected: List[Dict[str, Any]] = []
    for item in sessions:
        if not isinstance(item, dict):
            issues.append("selection_index_invalid_item")
            continue
        symptom, cause, variant = item.get("symptom"), item.get("cause"), item.get("variant")
        session = item.get("session")
        record_id = item.get("id")
        if not all(isinstance(value, str) for value in (symptom, cause, variant, session, record_id)):
            issues.append("selection_index_missing_identity")
            continue
        if record_id != f"{symptom}/{cause}/{variant}":
            issues.append(f"selection_index_id_mismatch:{record_id}")
            continue
        try:
            session_path = (root / session).resolve()
            session_path.relative_to(root.resolve())
        except ValueError:
            issues.append(f"selection_index_session_outside_repo:{record_id}")
            continue
        selected.append({"id": record_id, "symptom": symptom, "cause": cause, "variant": variant, "session": relative(session_path, root)})

    ids = [item["id"] for item in selected]
    if len(ids) != 30:
        issues.append(f"selection_count_expected_30_got_{len(ids)}")
    if len(set(ids)) != len(ids):
        issues.append("selection_duplicate_ids")
    missing = sorted(set(canonical_ids()) - set(ids))
    extra = sorted(set(ids) - set(canonical_ids()))
    if missing:
        issues.append("selection_missing:" + ",".join(missing))
    if extra:
        issues.append("selection_extra:" + ",".join(extra))
    return selected, issues, sha256(index_path)


def file_inventory(root: Path, session: Path) -> Dict[str, Dict[str, Any]]:
    inventory: Dict[str, Dict[str, Any]] = {}
    for role, name in REQUIRED_FILES.items():
        path = session / name
        entry: Dict[str, Any] = {"path": relative(path, root), "exists": path.is_file()}
        if path.is_file():
            stats = path.stat()
            entry.update({"bytes": stats.st_size, "mtime_utc": time_text(stats.st_mtime), "sha256": sha256(path)})
        inventory[role] = entry
    return inventory


def normalized_log_path(value: str) -> str:
    return value.strip().replace(CONTAINER_WORKSPACE, "").lstrip("./")


def log_events(log: str) -> Dict[str, Any]:
    event_patterns = {
        "session_start": re.compile(r"^(\S+) SESSION_START (.*)$", re.MULTILINE),
        "video_start": re.compile(r"^(\S+) VIDEO_START run=1 path=(.+)$", re.MULTILINE),
        "video_done": re.compile(r"^(\S+) VIDEO_DONE run=1 outputs=(.+)$", re.MULTILINE),
        "run_result": re.compile(r"^(\S+) RUN_RESULT run=1 success=(True|False) (.*)$", re.MULTILINE),
    }
    found: Dict[str, Any] = {}
    for name, pattern in event_patterns.items():
        matches = list(pattern.finditer(log))
        if matches:
            match = matches[-1]
            found[name] = {"timestamp": match.group(1), "details": match.group(2)}
    return found


def expected_config_matches(item: Mapping[str, str], config: Mapping[str, Any]) -> List[str]:
    symptom, cause, variant = item["symptom"], item["cause"], item["variant"]
    issues: List[str] = []
    if not config:
        return ["config_missing"]
    if cause == "shared-normal":
        for key in ("fault1_variant", "fault2_variant", "fault3_variant"):
            # Fault 1 records predate the Fault 2/3 fields. Omission means the
            # feature was unavailable and therefore in its normal state.
            if config.get(key, "normal") != "normal":
                issues.append(f"normal_config_{key}_is_{config.get(key)}")
        if config.get("midway_drop_cause") is not None or config.get("midway_drop_variant", "normal") != "normal":
            issues.append("normal_config_midway_drop_active")
        if config.get("misplacement_cause") is not None or config.get("misplacement_variant", "normal") != "normal":
            issues.append("normal_config_misplacement_active")
        return issues

    mechanism = SYMPTOMS[symptom]["causes"][cause][1]
    if mechanism.startswith("fault"):
        if config.get(mechanism, "normal") != variant:
            issues.append(f"config_{mechanism}_is_{config.get(mechanism)}_not_{variant}")
    elif mechanism == "midway_drop":
        if config.get("midway_drop_cause") != cause or config.get("midway_drop_variant") != variant:
            issues.append("config_midway_drop_label_mismatch")
    elif mechanism == "misplacement":
        if config.get("misplacement_cause") != cause or config.get("misplacement_variant") != variant:
            issues.append("config_misplacement_label_mismatch")
    for key in ("fault1_variant", "fault2_variant", "fault3_variant"):
        if mechanism != key and config.get(key, "normal") != "normal":
            issues.append(f"unrelated_config_{key}_is_{config.get(key)}")
    return issues


def expected_result_matches(item: Mapping[str, str], result: Mapping[str, Any]) -> List[str]:
    metrics = result.get("metrics") if isinstance(result, dict) else None
    if not isinstance(metrics, dict):
        return ["result_metrics_missing"]
    expected_success = item["variant"] in {"normal", "true-fix"}
    issues: List[str] = []
    if metrics.get("success") is not expected_success:
        issues.append(f"result_success_is_{metrics.get('success')}_not_{expected_success}")
    symptom, variant = item["symptom"], item["variant"]
    if symptom == "cannot-grasp" and not expected_success and metrics.get("lifted"):
        issues.append("cannot_grasp_label_but_cube_lifted")
    if symptom == "midway-drop":
        if not expected_success and not (metrics.get("grasp_established_before_fault") and metrics.get("drop_detected")):
            issues.append("midway_drop_label_missing_grasp_or_drop_evidence")
        if expected_success and metrics.get("drop_detected"):
            issues.append("midway_drop_recovery_still_reports_drop")
    if symptom == "inaccurate-placement":
        if not expected_success and not (
            metrics.get("lifted") and not metrics.get("drop_detected") and metrics.get("misplacement_detected")
        ):
            issues.append("misplacement_label_missing_transport_evidence")
        if expected_success and metrics.get("misplacement_detected"):
            issues.append("misplacement_recovery_still_reports_misplacement")
    return issues


def expected_log_matches(item: Mapping[str, str], config: Mapping[str, Any], events: Mapping[str, Any]) -> List[str]:
    start = events.get("session_start")
    if not start:
        return ["log_session_start_missing"]
    details = start["details"]
    symptom, cause, variant = item["symptom"], item["cause"], item["variant"]
    issues: List[str] = []
    if cause != "shared-normal":
        mechanism = SYMPTOMS[symptom]["causes"][cause][1]
        if mechanism.startswith("fault") and f"{mechanism}={variant}" not in details:
            issues.append("log_fault_variant_label_mismatch")
        if mechanism == "midway_drop" and (
            f"midway_drop_cause={cause}" not in details or f"midway_drop_variant={variant}" not in details
        ):
            issues.append("log_midway_drop_label_mismatch")
        if mechanism == "misplacement" and (
            f"misplacement_cause={cause}" not in details or f"misplacement_variant={variant}" not in details
        ):
            issues.append("log_misplacement_label_mismatch")
    else:
        if "fault=none" not in details:
            issues.append("log_normal_label_not_fault_none")
    run_result = events.get("run_result")
    expected_success = item["variant"] in {"normal", "true-fix"}
    if not run_result or (run_result.get("details") is not None and (run_result["details"] is None)):
        issues.append("log_run_result_missing")
    elif (run_result["details"] is not None) and (("success=True" in "RUN_RESULT run=1 success=" + run_result["details"]) != expected_success):
        issues.append("log_run_result_label_mismatch")
    return issues


def snapshot_record(root: Path, item: Mapping[str, str]) -> Dict[str, Any]:
    session = resolve_repo_path(root, item["session"])
    inventory = file_inventory(root, session)
    available = all(entry["exists"] for entry in inventory.values())
    summary_path = session / REQUIRED_FILES["session_summary"]
    result_path = session / REQUIRED_FILES["run_record"]
    trace_path = session / REQUIRED_FILES["trace"]
    log_path = session / REQUIRED_FILES["experiment_log"]
    video_path = session / REQUIRED_FILES["video"]
    summary, summary_error = json_file(summary_path) if summary_path.is_file() else (None, "missing")
    result, result_error = json_file(result_path) if result_path.is_file() else (None, "missing")
    trace_count = 0
    trace_error = None
    if trace_path.is_file():
        try:
            for trace_count, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines(), 1):
                json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            trace_error = f"{type(exc).__name__}:{exc}"
    else:
        trace_error = "missing"
    log = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    events = log_events(log)
    config = (summary or {}).get("config") if isinstance(summary, dict) else None
    metrics = (result or {}).get("metrics") if isinstance(result, dict) else None
    duration = mp4_duration_seconds(video_path) if video_path.is_file() else None
    timing = {
        "run_started_at": (result or {}).get("started_at"),
        "run_finished_at": (result or {}).get("finished_at"),
        "task_elapsed_wall_seconds": (result or {}).get("task_elapsed_wall_seconds"),
        "simulation_seconds_at_60hz": (result or {}).get("simulation_seconds_at_60hz"),
        "simulation_steps": (result or {}).get("simulation_steps"),
        "trace_observation_count": trace_count,
        "video_duration_seconds": duration,
        "video_mtime_utc": inventory["video"].get("mtime_utc"),
        "video_start_at": (events.get("video_start") or {}).get("timestamp"),
        "video_done_at": (events.get("video_done") or {}).get("timestamp"),
    }
    return {
        "id": item["id"],
        "source_session": item["session"],
        "labels": labels_for(item["symptom"], item["cause"], item["variant"]),
        "files": inventory,
        "run_conditions": config,
        "metrics": metrics,
        "timing": timing,
        "events": events,
        "summary": summary,
        "result": result,
        "parse_errors": {"summary": summary_error, "result": result_error, "trace": trace_error},
        "available": available,
    }


def add_issue(issues: Dict[str, List[str]], category: str, message: str) -> None:
    issues[category].append(message)


def validate_timing(snapshot: Mapping[str, Any]) -> List[str]:
    timing = snapshot["timing"]
    issues: List[str] = []
    run_start = parse_time(timing.get("run_started_at"))
    run_end = parse_time(timing.get("run_finished_at"))
    video_start = parse_time(timing.get("video_start_at"))
    video_done = parse_time(timing.get("video_done_at"))
    mtime = parse_time(timing.get("video_mtime_utc"))
    if not all((run_start, run_end, video_start, video_done, mtime)):
        return ["video_or_run_timestamp_missing"]
    assert run_start and run_end and video_start and video_done and mtime
    if not (run_start <= video_start <= video_done <= run_end + timedelta(seconds=TIMING_TOLERANCE_SECONDS)):
        issues.append("video_log_interval_not_within_run_interval")
    if not (video_start - timedelta(seconds=TIMING_TOLERANCE_SECONDS) <= mtime <= video_done + timedelta(seconds=TIMING_TOLERANCE_SECONDS)):
        issues.append("video_file_mtime_not_aligned_with_video_log")
    duration = timing.get("video_duration_seconds")
    simulation_seconds = timing.get("simulation_seconds_at_60hz")
    wall = (run_end - run_start).total_seconds()
    if not isinstance(duration, (int, float)) or duration <= 0:
        issues.append("mp4_duration_missing_or_nonpositive")
    else:
        if duration > wall + TIMING_TOLERANCE_SECONDS:
            issues.append("video_duration_exceeds_run_interval")
        if isinstance(simulation_seconds, (int, float)) and duration < simulation_seconds * 0.5:
            issues.append("video_duration_too_short_for_simulation")
    return issues


def validate_snapshot(root: Path, item: Mapping[str, str], snapshot: Dict[str, Any], baseline: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    issues: Dict[str, List[str]] = defaultdict(list)
    if not snapshot["available"]:
        for role, entry in snapshot["files"].items():
            if not entry["exists"]:
                add_issue(issues, "file_availability", f"missing:{role}")
    for name, error in snapshot["parse_errors"].items():
        if error:
            add_issue(issues, "run_record", f"{name}_parse:{error}")
    config = snapshot["run_conditions"]
    result = snapshot["result"] or {}
    for error in expected_config_matches(item, config or {}):
        add_issue(issues, "label_consistency", error)
    for error in expected_result_matches(item, result):
        add_issue(issues, "label_consistency", error)
    events = snapshot["events"]
    for error in expected_log_matches(item, config or {}, events):
        add_issue(issues, "label_video_consistency", error)
    expected_video = snapshot["files"]["video"]["path"]
    start_path = normalized_log_path((events.get("video_start") or {}).get("details", ""))
    done_details = (events.get("video_done") or {}).get("details", "")
    if start_path != expected_video:
        add_issue(issues, "label_video_consistency", "video_start_path_does_not_match_manifest")
    if expected_video not in done_details and str(Path(expected_video).name) not in done_details:
        add_issue(issues, "label_video_consistency", "video_done_path_does_not_match_manifest")
    for error in validate_timing(snapshot):
        add_issue(issues, "timing", error)
    trace_count = snapshot["timing"].get("trace_observation_count")
    steps = snapshot["timing"].get("simulation_steps")
    # One trace entry is captured immediately before the first simulated step.
    if isinstance(steps, int) and trace_count not in {steps, steps + 1}:
        add_issue(issues, "run_record", f"trace_count_{trace_count}_does_not_match_steps_{steps}_or_{steps + 1}")
    summary_runs = (snapshot["summary"] or {}).get("runs")
    metrics = snapshot["metrics"] or {}
    if not isinstance(summary_runs, list) or len(summary_runs) != 1:
        add_issue(issues, "run_record", "summary_does_not_contain_exactly_one_run")
    elif (summary_runs[0].get("metrics") or {}).get("success") != metrics.get("success"):
        add_issue(issues, "run_record", "summary_result_success_mismatch")

    if baseline is not None:
        if snapshot["source_session"] != baseline.get("source_session"):
            add_issue(issues, "integrity", "source_session_changed")
        if snapshot["labels"] != baseline.get("labels"):
            add_issue(issues, "integrity", "labels_changed")
        for role, expected in (baseline.get("files") or {}).items():
            actual = snapshot["files"].get(role, {})
            if not actual.get("exists"):
                continue
            if actual.get("path") != expected.get("path"):
                add_issue(issues, "integrity", f"file_path_changed:{role}")
            if actual.get("sha256") != expected.get("sha256"):
                add_issue(issues, "integrity", f"file_hash_changed:{role}")
            elif actual.get("mtime_utc") != expected.get("mtime_utc"):
                add_issue(issues, "integrity", f"file_mtime_changed:{role}")
    checks = {
        "file_availability": not issues["file_availability"],
        "run_record": not issues["run_record"],
        "label_consistency": not issues["label_consistency"],
        "label_video_consistency": not issues["label_video_consistency"],
        "timing": not issues["timing"],
        "integrity": not issues["integrity"],
    }
    nonempty_issues = {category: messages for category, messages in issues.items() if messages}
    snapshot["quality"] = {"passed": all(checks.values()), "checks": checks, "issues": nonempty_issues}
    return snapshot


def completeness_check(selected: Iterable[Mapping[str, str]]) -> Dict[str, Any]:
    ids = {item["id"] for item in selected}
    expected = set(canonical_ids())
    by_symptom: Dict[str, Any] = {}
    for symptom, spec in SYMPTOMS.items():
        normal = f"{symptom}/shared-normal/normal" in ids
        causes = {}
        for cause in spec["causes"]:
            causes[cause] = {variant: f"{symptom}/{cause}/{variant}" in ids for variant in VARIANTS[1:]}
        by_symptom[symptom] = {
            "shared_normal_present": normal,
            "all_four_variant_classes_covered": normal and all(all(values.values()) for values in causes.values()),
            "causes": causes,
        }
    return {
        "expected_session_count": 30,
        "actual_session_count": len(ids),
        "missing_ids": sorted(expected - ids),
        "unexpected_ids": sorted(ids - expected),
        "symptoms": by_symptom,
        "passed": ids == expected,
    }


def record_filename(record_id: str) -> str:
    return record_id.replace("/", "--") + ".json"


def write_record_cards(root: Path, output: Path, records: Iterable[Mapping[str, Any]], reference_name: str) -> None:
    for record in records:
        card = {
            "schema": "calibration-record-card-v1",
            "record_id": record["id"],
            "source_session": record["source_session"],
            "labels": record["labels"],
            "run_conditions": record["run_conditions"],
            "run_result": {
                "metrics": record["metrics"],
                "phase_history": (record["result"] or {}).get("phase_history"),
                "timing": record["timing"],
            },
            "file_manifest": record["files"],
            "quality": record["quality"],
            "reference_manifest": reference_name,
        }
        write_json(output / "records" / record_filename(record["id"]), card)


def write_dataset_index(root: Path, output: Path, records: Iterable[Mapping[str, Any]], completeness: Mapping[str, Any]) -> None:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["labels"]["symptom"]].append(record)
    lines = [
        "# 可复查校准数据索引",
        "",
        "此索引固定选择 30 条人工检查通过的运行。每条均提供标签、实际运行条件、结果、轨迹、视频、日志和逐文件 SHA-256 清单；视频文件未被复制。",
        "",
    ]
    for symptom, spec in SYMPTOMS.items():
        lines.extend(
            [
                f"## {spec['label']} (`{symptom}`)",
                "",
                "| 编号 | 原因 | 版本 | 期望结果 | 本次结果 | 视频 | 记录卡 |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for number, record in enumerate(grouped[symptom], 1):
            labels = record["labels"]
            video = output_link(record["files"]["video"]["path"], root, output)
            card = "records/" + record_filename(record["id"])
            success = (record["metrics"] or {}).get("success")
            lines.append(
                f"| {number} | {labels['cause_label']} | `{labels['variant']}` | `{labels['expected_success']}` | "
                f"`{success}` | [MP4]({video}) | [标签、条件、清单]({card}) |"
            )
        complete = completeness["symptoms"][symptom]
        lines.extend(
            [
                "",
                f"- 共享正常基线：**{complete['shared_normal_present']}**；每个原因均具备故障、错误归因、正确处理：**{complete['all_four_variant_classes_covered']}**。",
                "",
            ]
        )
    lines.extend(
        [
            "## 复现信息",
            "",
            "每条记录卡保存完整 `run_conditions`，包括随机种子、设备、IK、稳定帧数、立方体初始位置、目标位置、阶段步数和容差。",
            "质量系统的冻结文件清单在 [reference-manifest.json](reference-manifest.json)，逐次检查结果在 [quality-report.md](quality-report.md)。",
        ]
    )
    (output / "dataset-index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_report(output: Path, records: Iterable[Mapping[str, Any]], completeness: Mapping[str, Any], mode: str, reference_ok: bool) -> Dict[str, Any]:
    listed = list(records)
    issue_counts: Counter[str] = Counter()
    failures: List[Dict[str, Any]] = []
    for record in listed:
        for category, issues in record["quality"]["issues"].items():
            issue_counts[category] += len(issues)
        if not record["quality"]["passed"]:
            failures.append({"id": record["id"], "issues": record["quality"]["issues"]})
    report = {
        "schema": "calibration-quality-report-v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "record_count": len(listed),
        "tracked_file_count": sum(len(record["files"]) for record in listed),
        "reference_manifest_checksum_valid": reference_ok,
        "variant_completeness": completeness,
        "issue_counts": dict(issue_counts),
        "failed_records": failures,
        "passed": not failures and completeness["passed"] and reference_ok,
    }
    write_json(output / "quality-report.json", report)
    lines = [
        "# 校准数据质量报告",
        "",
        f"- 检查模式：`{mode}`",
        f"- 会话数：**{len(listed)}/30**；受保护文件数：**{report['tracked_file_count']}**。",
        f"- 四类版本完整性：**{completeness['passed']}**；参考清单校验：**{reference_ok}**。",
        f"- 总体结果：**{'通过' if report['passed'] else '失败'}**。",
        "",
        "## 自动检查项",
        "",
        "| 项目 | 方法 | 结果 |",
        "| --- | --- | --- |",
        f"| 文件缺失 | 每条要求 MP4、结果、轨迹、实验日志、会话摘要 | {'通过' if not issue_counts['file_availability'] else '失败'} |",
        f"| 标签与运行条件 | 标签与 `summary.json`、`result.json` 的原因、版本、成功状态及症状指标交叉验证 | {'通过' if not issue_counts['label_consistency'] else '失败'} |",
        f"| 标签与视频绑定 | `VIDEO_START`/`VIDEO_DONE` 必须指向同一受管 MP4，日志版本和 `RUN_RESULT` 必须匹配标签 | {'通过' if not issue_counts['label_video_consistency'] else '失败'} |",
        f"| 视频时间对齐 | 视频起止日志、MP4 修改时间、运行起止时间和 MP4 容器时长交叉验证 | {'通过' if not issue_counts['timing'] else '失败'} |",
        f"| 四类版本完整性 | 每种症状 1 条共享正常基线 + 3 个原因各有故障/错误归因/正确处理 | {'通过' if completeness['passed'] else '失败'} |",
        f"| 文件被修改 | 每个受保护文件的路径、SHA-256、大小和修改时间同冻结基线比较 | {'通过' if not issue_counts['integrity'] else '失败'} |",
        "",
    ]
    if failures:
        lines.extend(["## 发现的问题", ""])
        for failure in failures:
            joined = "; ".join(f"{category}: {', '.join(messages)}" for category, messages in failure["issues"].items())
            lines.append(f"- `{failure['id']}`: {joined}")
    else:
        lines.extend(["## 结论", "", "30 条既有校准数据均可追溯，文件清单、标签、运行条件、结果和视频绑定一致。", ""])
    (output / "quality-report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "calibration-index.json"
    reference_path = output / "reference-manifest.json"
    checksum_path = output / "reference-manifest.sha256"
    mode = "initialize" if args.initialize else "verify"
    reference_ok = True
    selection_issues: List[str] = []
    baseline_by_id: Dict[str, Mapping[str, Any]] = {}

    if args.initialize:
        if reference_path.exists() or checksum_path.exists():
            print("Refusing to overwrite an existing reference manifest. Use --verify for future checks.", file=sys.stderr)
            return 2
        selected, selection_issues, index_hash = load_selection(root, index_path)
    else:
        reference, reference_error = json_file(reference_path)
        if reference_error or reference is None:
            print(f"Reference manifest unavailable: {reference_error}", file=sys.stderr)
            return 2
        expected_hash = checksum_path.read_text(encoding="utf-8").strip().split()[0] if checksum_path.is_file() else ""
        reference_ok = bool(expected_hash) and sha256(reference_path) == expected_hash
        if not reference_ok:
            selection_issues.append("reference_manifest_checksum_invalid")
        raw_records = reference.get("records")
        if not isinstance(raw_records, list):
            print("Reference manifest records missing.", file=sys.stderr)
            return 2
        selected = []
        for entry in raw_records:
            labels = entry.get("labels") or {}
            record_id = entry.get("id")
            if not isinstance(record_id, str) or not isinstance(entry.get("source_session"), str):
                selection_issues.append("reference_manifest_invalid_record")
                continue
            selected.append(
                {
                    "id": record_id,
                    "symptom": labels.get("symptom"),
                    "cause": labels.get("cause"),
                    "variant": labels.get("variant"),
                    "session": entry["source_session"],
                }
            )
            baseline_by_id[record_id] = entry
        index_hash = (reference.get("source_selection") or {}).get("sha256", "")

    completeness = completeness_check(selected)
    selection_issues.extend(completeness["missing_ids"])
    selection_issues.extend(completeness["unexpected_ids"])
    records: List[Dict[str, Any]] = []
    for item in selected:
        if item["symptom"] not in SYMPTOMS or item["variant"] not in VARIANTS:
            selection_issues.append(f"invalid_identity:{item.get('id')}")
            continue
        snapshot = snapshot_record(root, item)
        records.append(validate_snapshot(root, item, snapshot, baseline_by_id.get(item["id"])))
    if selection_issues:
        for record in records:
            for issue in selection_issues:
                add_issue(record["quality"]["issues"], "variant_completeness", issue)
            record["quality"]["checks"]["variant_completeness"] = False
            record["quality"]["passed"] = False

    if args.initialize:
        preflight = write_quality_report(output, records, completeness, mode, reference_ok)
        if not preflight["passed"]:
            print("Reference manifest was not created because the preflight audit failed.", file=sys.stderr)
            return 1
        reference = {
            "schema": "calibration-reference-manifest-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_selection": {"path": relative(index_path, root), "sha256": index_hash},
            "selection_rule": "one shared normal plus three variants for each of three single-cause interventions",
            "records": [
                {
                    "id": record["id"],
                    "source_session": record["source_session"],
                    "labels": record["labels"],
                    "run_conditions": record["run_conditions"],
                    "files": record["files"],
                    "timing": record["timing"],
                }
                for record in records
            ],
        }
        write_json(reference_path, reference)
        checksum_path.write_text(f"{sha256(reference_path)}  reference-manifest.json\n", encoding="utf-8")
        write_json(output / "file-manifest.json", {"schema": "calibration-file-manifest-v1", "records": reference["records"]})
        mode = "initialize"

    reference_name = "reference-manifest.json"
    write_record_cards(root, output, records, reference_name)
    write_dataset_index(root, output, records, completeness)
    report = write_quality_report(output, records, completeness, mode, reference_ok)
    print(f"Wrote dataset index to {output / 'dataset-index.md'}")
    print(f"Wrote quality report to {output / 'quality-report.md'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
