"""Build and verify the aggregate index for the ten-condition calibration set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from audit_calibration_dataset import canonical_ids, sha256, write_json


CONTAINER_ROOT = Path("/workspace/TwinTicket-Sim")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", type=Path, default=Path("calibration_scaleup_plan.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration-300"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            return root / candidate.relative_to(CONTAINER_ROOT)
        except ValueError:
            return candidate
    return candidate if candidate.is_absolute() else root / candidate


def read_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def reference_integrity(root: Path, dataset: Path) -> Tuple[bool, List[str], Optional[str]]:
    manifest_path = dataset / "reference-manifest.json"
    checksum_path = dataset / "reference-manifest.sha256"
    issues: List[str] = []
    if not manifest_path.is_file() or not checksum_path.is_file():
        return False, ["reference_manifest_missing"], None
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    actual = sha256(manifest_path)
    if actual != expected:
        issues.append("reference_manifest_checksum_mismatch")
        return False, issues, actual
    manifest = read_json(manifest_path)
    file_entries: List[Tuple[str, Mapping[str, Any]]] = []
    for record in manifest.get("records", []):
        for role, entry in (record.get("files") or {}).items():
            file_entries.append((f"record:{record.get('id')}:{role}", entry))
    for attempt in manifest.get("attempts", []):
        for role, entry in (attempt.get("files") or {}).items():
            file_entries.append(
                (f"attempt:{attempt.get('id')}:{attempt.get('attempt')}:{role}", entry)
            )
    for identity, entry in file_entries:
        path_value = entry.get("path")
        if not isinstance(path_value, str):
            issues.append(f"manifest_path_missing:{identity}")
            continue
        path = resolve(root, path_value)
        if not path.is_file():
            issues.append(f"source_file_missing:{identity}")
        elif sha256(path) != entry.get("sha256"):
            issues.append(f"source_file_hash_changed:{identity}")
    return not issues, issues, actual


def default_condition(
    root: Path, condition: Mapping[str, Any], dataset: Path
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    issues: List[str] = []
    try:
        quality = read_json(dataset / "quality-report.json")
        index = read_json(dataset / "calibration-index.json")
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        return {}, [], [f"canonical_dataset_unreadable:{exc}"]
    records = []
    for item in index.get("sessions", []):
        metrics = item.get("metrics") or {}
        records.append(
            {
                "condition_id": condition["id"],
                "cube_initial_position_m": condition["cube_initial_position_m"],
                "id": item.get("id"),
                "symptom": item.get("symptom"),
                "cause": item.get("cause"),
                "variant": item.get("variant"),
                "session": item.get("session"),
                "video": str(Path(item.get("session", "")) / "run-001.mp4"),
                "actual_success": metrics.get("success"),
            }
        )
    integrity_ok, integrity_issues, manifest_hash = reference_integrity(root, dataset)
    issues.extend(integrity_issues)
    if quality.get("passed") is not True:
        issues.append("canonical_quality_report_failed")
    summary = {
        "id": condition["id"],
        "source": condition["source"],
        "dataset_dir": relative(dataset, root),
        "cube_initial_position_m": condition["cube_initial_position_m"],
        "status": "passed" if quality.get("passed") is True and integrity_ok and len(records) == 30 else "blocked",
        "record_count": len(records),
        "successful_outcomes": sum(item["actual_success"] is True for item in records),
        "failed_outcomes": sum(item["actual_success"] is False for item in records),
        "attempt_history_complete": False,
        "attempt_counts": None,
        "reference_manifest_sha256": manifest_hash,
        "issues": issues,
    }
    return summary, records, issues


def batch_condition(
    root: Path, condition: Mapping[str, Any], dataset: Path
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    issues: List[str] = []
    if not (dataset / "batch-state.json").is_file():
        summary = {
            "id": condition["id"],
            "source": condition["source"],
            "dataset_dir": relative(dataset, root),
            "cube_initial_position_m": condition["cube_initial_position_m"],
            "status": "not-started",
            "record_count": 0,
            "successful_outcomes": 0,
            "failed_outcomes": 0,
            "attempt_history_complete": True,
            "attempt_counts": None,
            "reference_manifest_sha256": None,
            "issues": [],
        }
        return summary, [], []
    try:
        state = read_json(dataset / "batch-state.json")
        index = read_json(dataset / "batch-index.json")
        quality = read_json(dataset / "quality-report.json") if (dataset / "quality-report.json").is_file() else {}
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        return {}, [], [f"batch_dataset_unreadable:{exc}"]
    if state.get("start_condition", {}).get("id") != condition["id"]:
        issues.append("start_condition_id_mismatch")
    if state.get("start_condition", {}).get("cube_initial_position_m") != condition["cube_initial_position_m"]:
        issues.append("start_condition_position_mismatch")
    records = []
    slots = state.get("slots") or {}
    for item in index.get("records", []):
        accepted = (slots.get(item.get("id")) or {}).get("accepted_attempt") or {}
        metrics = accepted.get("metrics") or {}
        session = resolve(root, item.get("session", ""))
        records.append(
            {
                "condition_id": condition["id"],
                "cube_initial_position_m": condition["cube_initial_position_m"],
                "id": item.get("id"),
                "symptom": item.get("symptom"),
                "cause": item.get("cause"),
                "variant": item.get("variant"),
                "session": relative(session, root),
                "video": relative(session / "run-001.mp4", root),
                "actual_success": metrics.get("success"),
                "attempt": item.get("attempt"),
            }
        )
    integrity_ok = False
    manifest_hash = None
    if (dataset / "reference-manifest.json").is_file():
        integrity_ok, integrity_issues, manifest_hash = reference_integrity(root, dataset)
        issues.extend(integrity_issues)
    elif len(records) == 30:
        issues.append("completed_batch_reference_missing")
    quality_passed = quality.get("passed") is True
    if len(records) == 30 and not quality_passed:
        issues.append("batch_quality_report_failed")
    if len(records) < 30 and state.get("condition_decision", {}).get("accepted") is False:
        status = "blocked"
    elif len(records) == 30 and quality_passed and integrity_ok:
        status = "passed"
    elif records:
        status = "in-progress"
    else:
        status = "started"
    counts = state.get("counts") or {}
    summary = {
        "id": condition["id"],
        "source": condition["source"],
        "dataset_dir": relative(dataset, root),
        "cube_initial_position_m": condition["cube_initial_position_m"],
        "status": status,
        "record_count": len(records),
        "successful_outcomes": sum(item["actual_success"] is True for item in records),
        "failed_outcomes": sum(item["actual_success"] is False for item in records),
        "attempt_history_complete": True,
        "attempt_counts": counts,
        "reference_manifest_sha256": manifest_hash,
        "issues": issues,
    }
    return summary, records, issues


def validate_plan(plan: Mapping[str, Any]) -> List[str]:
    issues: List[str] = []
    conditions = plan.get("conditions") or []
    ids = [item.get("id") for item in conditions]
    positions = [tuple(item.get("cube_initial_position_m") or []) for item in conditions]
    if len(conditions) != 10:
        issues.append(f"condition_count_is_{len(conditions)}_not_10")
    if len(set(ids)) != len(ids):
        issues.append("duplicate_condition_id")
    if len(set(positions)) != len(positions):
        issues.append("duplicate_start_position")
    for item in conditions:
        if item.get("source") not in {"canonical-existing", "validated-existing", "generated"}:
            issues.append(f"invalid_condition_source:{item.get('id')}")
        if len(item.get("cube_initial_position_m") or []) != 3:
            issues.append(f"invalid_start_position:{item.get('id')}")
    return issues


def sum_attempt_counts(conditions: Iterable[Mapping[str, Any]], source: Optional[str] = None) -> Dict[str, Any]:
    keys = (
        "total_attempts",
        "attempt_successful_outcomes",
        "attempt_failed_outcomes",
        "attempts_without_outcome",
        "rerun_attempts",
        "void_attempts",
    )
    selected = [
        item
        for item in conditions
        if item.get("attempt_history_complete") and item.get("attempt_counts")
        and (source is None or item.get("source") == source)
    ]
    return {
        **{key: sum(int((item["attempt_counts"] or {}).get(key, 0)) for item in selected) for key in keys},
        "condition_count": len(selected),
    }


def excluded_evidence_summaries(
    root: Path, excluded: Iterable[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    summaries: List[Dict[str, Any]] = []
    issues: List[str] = []
    for item in excluded:
        dataset = resolve(root, item["dataset_dir"])
        state_path = dataset / "batch-state.json"
        summary: Dict[str, Any] = {
            "id": item["id"],
            "cube_initial_position_m": item["cube_initial_position_m"],
            "dataset_dir": relative(dataset, root),
            "reason": item["reason"],
            "attempt_counts": None,
            "evidence_file_count": 0,
            "evidence_complete": False,
        }
        if state_path.is_file():
            try:
                state = read_json(state_path)
            except (OSError, json.JSONDecodeError, RuntimeError) as exc:
                issues.append(f"excluded_evidence_unreadable:{item['id']}:{exc}")
            else:
                actual = state.get("start_condition") or {}
                if actual.get("id") != item["id"]:
                    issues.append(f"excluded_condition_id_mismatch:{item['id']}")
                if actual.get("cube_initial_position_m") != item["cube_initial_position_m"]:
                    issues.append(f"excluded_condition_position_mismatch:{item['id']}")
                summary["attempt_counts"] = state.get("counts") or {}
                tracked = 0
                complete = True
                for slot in (state.get("slots") or {}).values():
                    for attempt in slot.get("attempts", []):
                        sources = list((attempt.get("files") or {}).values())
                        if attempt.get("launcher_log"):
                            sources.append(attempt["launcher_log"])
                        for source in sources:
                            path_value = source.get("path")
                            path = resolve(root, path_value) if isinstance(path_value, str) else None
                            tracked += 1
                            if path is None or not path.is_file() or sha256(path) != source.get("sha256"):
                                complete = False
                summary["evidence_file_count"] = tracked
                summary["evidence_complete"] = complete and tracked > 0
                if tracked and not summary["evidence_complete"]:
                    issues.append(f"excluded_evidence_file_changed:{item['id']}")
        summaries.append(summary)
    return summaries, issues


def write_outputs(
    root: Path,
    output: Path,
    plan: Mapping[str, Any],
    condition_summaries: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    issues: List[str],
    mode: str,
    reference_ok: bool,
    excluded_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    composites = [(item["condition_id"], item["id"]) for item in records]
    if len(set(composites)) != len(composites):
        issues.append("duplicate_condition_record_identity")
    sessions = [item.get("session") for item in records if item.get("session")]
    if len(set(sessions)) != len(sessions):
        issues.append("duplicate_source_session")
    expected_ids = set(canonical_ids())
    for condition in condition_summaries:
        ids = {item["id"] for item in records if item["condition_id"] == condition["id"]}
        if condition["status"] == "passed" and ids != expected_ids:
            issues.append(f"canonical_slot_set_mismatch:{condition['id']}")
    symptom_counts = Counter(item["symptom"] for item in records)
    all_passed = len(condition_summaries) == 10 and all(item.get("status") == "passed" for item in condition_summaries)
    if len(records) == 300:
        for symptom in ("cannot-grasp", "midway-drop", "inaccurate-placement"):
            if symptom_counts[symptom] != 100:
                issues.append(f"symptom_count_{symptom}_is_{symptom_counts[symptom]}_not_100")
    accepted_successes = sum(item.get("actual_success") is True for item in records)
    accepted_failures = sum(item.get("actual_success") is False for item in records)
    passed = all_passed and len(records) == 300 and accepted_successes == 120 and accepted_failures == 180 and not issues and reference_ok
    tracked_attempts = sum_attempt_counts(condition_summaries)
    generated_attempts = sum_attempt_counts(condition_summaries, "generated")
    excluded_attempts = sum_attempt_counts(
        [
            {
                "attempt_history_complete": item.get("attempt_counts") is not None,
                "attempt_counts": item.get("attempt_counts"),
                "source": "excluded",
            }
            for item in excluded_summaries
        ]
    )
    all_recorded_attempts = {
        key: tracked_attempts[key] + excluded_attempts[key]
        for key in (
            "total_attempts",
            "attempt_successful_outcomes",
            "attempt_failed_outcomes",
            "attempts_without_outcome",
            "rerun_attempts",
            "void_attempts",
        )
    }
    counts = {
        "target_records": 300,
        "accepted_records": len(records),
        "accepted_successful_outcomes": accepted_successes,
        "accepted_failed_outcomes": accepted_failures,
        "passed_conditions": sum(item.get("status") == "passed" for item in condition_summaries),
        "blocked_conditions": sum(item.get("status") == "blocked" for item in condition_summaries),
        "not_started_conditions": sum(item.get("status") == "not-started" for item in condition_summaries),
        "tracked_attempts_excluding_incomplete_canonical_history": tracked_attempts,
        "generated_this_scaleup": generated_attempts,
        "retired_condition_attempts": excluded_attempts,
        "all_recorded_attempts_excluding_incomplete_canonical_history": all_recorded_attempts,
        "attempt_history_unknown_conditions": [
            item["id"] for item in condition_summaries if not item.get("attempt_history_complete")
        ],
    }
    index = {
        "schema": "calibration-scaleup-index-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": sha256(resolve(root, "calibration_scaleup_plan.json")),
        "counts": counts,
        "conditions": condition_summaries,
        "records": records,
        "excluded_evidence": plan.get("excluded_evidence") or [],
        "excluded_evidence_summaries": excluded_summaries,
    }
    write_json(output / "scaleup-index.json", index)
    report = {
        "schema": "calibration-scaleup-quality-report-v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "counts": counts,
        "symptom_counts": dict(symptom_counts),
        "condition_summaries": condition_summaries,
        "issues": sorted(set(issues)),
        "aggregate_reference_valid": reference_ok,
        "passed": passed,
    }
    write_json(output / "quality-report.json", report)

    index_lines = [
        "# 300 条校准数据索引",
        "",
        f"当前有效数据 **{len(records)}/300** 条，开始条件 **{counts['passed_conditions']}/10** 个通过。",
        "",
        "| 开始条件 | 方块初始位置 | 来源 | 有效条数 | 成功/失败 | 状态 | 子批次索引 |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in condition_summaries:
        dataset = item["dataset_dir"]
        child = "dataset-index.md" if item["source"] == "canonical-existing" else "batch-index.md"
        link = Path("..") / Path(dataset).relative_to("artifacts") / child
        index_lines.append(
            f"| `{item['id']}` | `{item['cube_initial_position_m']}` | `{item['source']}` | "
            f"{item['record_count']} | {item['successful_outcomes']}/{item['failed_outcomes']} | "
            f"`{item['status']}` | [查看]({link}) |"
        )
    index_lines.extend([
        "",
        "## 逐条记录",
        "",
        "| 开始条件 | 数据 ID | 实际结果 | 视频 |",
        "| --- | --- | --- | --- |",
    ])
    for item in records:
        video = Path("..") / Path(item["video"]).relative_to("artifacts")
        index_lines.append(
            f"| `{item['condition_id']}` | `{item['id']}` | `{item['actual_success']}` | [MP4]({video}) |"
        )
    (output / "scaleup-index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    generated = counts["generated_this_scaleup"]
    tracked = counts["tracked_attempts_excluding_incomplete_canonical_history"]
    report_lines = [
        "# 正式规模质量报告",
        "",
        f"- 质量结果：**{'通过' if passed else '未通过或未完成'}**。",
        f"- 有效数据：**{len(records)}/300**；通过开始条件：**{counts['passed_conditions']}/10**。",
        f"- 有效结果中的实际成功：**{accepted_successes}**；实际失败：**{accepted_failures}**。",
        f"- 本次新增条件实际尝试：**{generated['total_attempts']}**；成功：**{generated['attempt_successful_outcomes']}**；失败：**{generated['attempt_failed_outcomes']}**；无结果：**{generated['attempts_without_outcome']}**；重跑：**{generated['rerun_attempts']}**；作废：**{generated['void_attempts']}**。",
        f"- 有完整尝试历史的条件合计：实际尝试 **{tracked['total_attempts']}**；重跑 **{tracked['rerun_attempts']}**；作废 **{tracked['void_attempts']}**。",
        f"- 已淘汰条件证据：实际尝试 **{excluded_attempts['total_attempts']}**；成功 **{excluded_attempts['attempt_successful_outcomes']}**；失败 **{excluded_attempts['attempt_failed_outcomes']}**；重跑 **{excluded_attempts['rerun_attempts']}**；作废 **{excluded_attempts['void_attempts']}**。",
        f"- 所有有记录尝试（不含默认条件未知历史）：实际尝试 **{all_recorded_attempts['total_attempts']}**；成功 **{all_recorded_attempts['attempt_successful_outcomes']}**；失败 **{all_recorded_attempts['attempt_failed_outcomes']}**；重跑 **{all_recorded_attempts['rerun_attempts']}**；作废 **{all_recorded_attempts['void_attempts']}**。",
        "- 默认条件的 30 条历史数据只保留被选中的实际结果，早期被舍弃尝试没有完整记录，因此其历史重跑和作废数标记为未知，不以 0 代替。",
        "",
        "| 开始条件 | 状态 | 有效条数 | 实际尝试 | 重跑 | 作废 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in condition_summaries:
        attempt = item.get("attempt_counts") or {}
        report_lines.append(
            f"| `{item['id']}` | `{item['status']}` | {item['record_count']} | "
            f"{attempt.get('total_attempts', '未知')} | {attempt.get('rerun_attempts', '未知')} | "
            f"{attempt.get('void_attempts', '未知')} |"
        )
    if excluded_summaries:
        report_lines.extend([
            "",
            "## 已淘汰条件",
            "",
            "| 开始条件 | 初始位置 | 实际尝试 | 重跑 | 作废 | 证据 | 原因 |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ])
        for item in excluded_summaries:
            attempt = item.get("attempt_counts") or {}
            report_lines.append(
                f"| `{item['id']}` | `{item['cube_initial_position_m']}` | "
                f"{attempt.get('total_attempts', '未知')} | {attempt.get('rerun_attempts', '未知')} | "
                f"{attempt.get('void_attempts', '未知')} | "
                f"`{'完整' if item.get('evidence_complete') else '无运行或不完整'}` | {item['reason']} |"
            )
    if issues:
        report_lines.extend(["", "## 阻断问题", ""])
        report_lines.extend(f"- `{issue}`" for issue in sorted(set(issues)))
    (output / "quality-report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    plan_path = resolve(root, args.plan)
    output = resolve(root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mode = "initialize" if args.initialize else "verify" if args.verify else "status"
    try:
        plan = read_json(plan_path)
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Cannot read scale-up plan: {exc}", file=sys.stderr)
        return 2
    issues = validate_plan(plan)
    condition_summaries: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    excluded_summaries, excluded_issues = excluded_evidence_summaries(
        root, plan.get("excluded_evidence") or []
    )
    issues.extend(excluded_issues)
    for condition in plan.get("conditions", []):
        dataset = resolve(root, condition["dataset_dir"])
        if condition["source"] == "canonical-existing":
            summary, selected, child_issues = default_condition(root, condition, dataset)
        else:
            summary, selected, child_issues = batch_condition(root, condition, dataset)
        if summary:
            condition_summaries.append(summary)
        records.extend(selected)
        issues.extend(f"{condition['id']}:{issue}" for issue in child_issues)

    reference_path = output / "aggregate-reference-manifest.json"
    checksum_path = output / "aggregate-reference-manifest.sha256"
    reference_ok = True
    baseline: Optional[Dict[str, Any]] = None
    if args.initialize and (reference_path.exists() or checksum_path.exists()):
        print("Refusing to overwrite aggregate reference; use --verify.", file=sys.stderr)
        return 2
    if args.verify:
        if not reference_path.is_file() or not checksum_path.is_file():
            print("Aggregate reference is missing; use --initialize after all 300 records pass.", file=sys.stderr)
            return 2
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        reference_ok = sha256(reference_path) == expected
        baseline = read_json(reference_path)
        current_identity = [
            {
                "condition_id": item["condition_id"],
                "id": item["id"],
                "session": item["session"],
                "actual_success": item["actual_success"],
            }
            for item in records
        ]
        if current_identity != baseline.get("records"):
            issues.append("aggregate_record_selection_changed")
        current_conditions = [
            {
                "id": item["id"],
                "cube_initial_position_m": item["cube_initial_position_m"],
                "reference_manifest_sha256": item["reference_manifest_sha256"],
                "attempt_counts": item["attempt_counts"],
            }
            for item in condition_summaries
        ]
        if current_conditions != baseline.get("conditions"):
            issues.append("aggregate_condition_evidence_changed")
    report = write_outputs(
        root, output, plan, condition_summaries, records, issues, mode, reference_ok,
        excluded_summaries,
    )
    if args.initialize:
        if not report["passed"]:
            print("Aggregate reference was not initialized because the dataset is incomplete or failed quality checks.", file=sys.stderr)
            return 1
        reference = {
            "schema": "calibration-scaleup-reference-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": sha256(plan_path),
            "records": [
                {
                    "condition_id": item["condition_id"],
                    "id": item["id"],
                    "session": item["session"],
                    "actual_success": item["actual_success"],
                }
                for item in records
            ],
            "conditions": [
                {
                    "id": item["id"],
                    "cube_initial_position_m": item["cube_initial_position_m"],
                    "reference_manifest_sha256": item["reference_manifest_sha256"],
                    "attempt_counts": item["attempt_counts"],
                }
                for item in condition_summaries
            ],
        }
        write_json(reference_path, reference)
        checksum_path.write_text(
            f"{sha256(reference_path)}  aggregate-reference-manifest.json\n", encoding="utf-8"
        )
    print(f"Wrote aggregate scale-up report to {output / 'quality-report.md'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
