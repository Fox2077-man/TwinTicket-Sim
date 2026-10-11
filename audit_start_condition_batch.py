"""Initialize or verify quality metadata for the bounded start-condition batch."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from audit_calibration_dataset import (
    SYMPTOMS,
    completeness_check,
    record_filename,
    resolve_repo_path,
    sha256,
    snapshot_record,
    validate_snapshot,
    write_json,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("artifacts/start-condition-validation/cube-x-minus-0.01-y-plus-0.02"),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--initialize", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument(
        "--status",
        action="store_true",
        help="Check a complete or blocked batch without creating a frozen reference",
    )
    mode.add_argument(
        "--refresh",
        action="store_true",
        help="Archive the previous reference and replace it after a passing repair",
    )
    return parser.parse_args()


def load_batch_selection(root: Path, batch: Path) -> List[Dict[str, str]]:
    index = json.loads((batch / "batch-index.json").read_text(encoding="utf-8"))
    records = index.get("records")
    if not isinstance(records, list):
        raise RuntimeError("batch-index.json does not contain records")
    selected = []
    for record in records:
        selected.append(
            {
                "id": record["id"],
                "symptom": record["symptom"],
                "cause": record["cause"],
                "variant": record["variant"],
                "session": record["session"],
            }
        )
    return selected


def start_condition_issues(record: Mapping[str, Any], start_condition: Mapping[str, Any]) -> List[str]:
    config = record.get("run_conditions") or {}
    contract = config.get("task_contract") or {}
    issues = []
    if config.get("start_condition_id") != start_condition["id"]:
        issues.append("start_condition_id_mismatch")
    if contract.get("cube_initial_position_m") != start_condition["cube_initial_position_m"]:
        issues.append("cube_initial_position_mismatch")
    if record.get("id") == "midway-drop/motion-disturbance/wrong-cause":
        fault = config.get("fault") or {}
        metrics = record.get("metrics") or {}
        if fault.get("true_cause") != "downward_disturbance_velocity":
            issues.append("motion_true_cause_missing")
        if fault.get("diagnosis") != "gripper_command":
            issues.append("motion_wrong_diagnosis_mismatch")
        if fault.get("diagnosis_action") != "verify_close_command":
            issues.append("motion_wrong_action_changed_physics")
        if fault.get("true_cause_preserved") is not True:
            issues.append("motion_true_cause_not_preserved")
        if metrics.get("drop_disturbance_velocity_z_mps") != -3.0:
            issues.append("motion_disturbance_magnitude_changed")
        if metrics.get("midway_wrong_diagnosis") != "gripper_command":
            issues.append("motion_result_wrong_diagnosis_mismatch")
        if metrics.get("midway_wrong_diagnosis_action") != "verify_close_command":
            issues.append("motion_result_wrong_action_mismatch")
    return issues


def attempt_reference(root: Path, state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    attempts = []
    for slot_id, slot in state.get("slots", {}).items():
        for attempt in slot.get("attempts", []):
            files = {}
            sources = dict(attempt.get("files") or {})
            if attempt.get("launcher_log"):
                sources["launcher.log"] = attempt["launcher_log"]
            for role, source in sources.items():
                path = resolve_repo_path(root, source["path"])
                files[role] = {
                    "path": source["path"],
                    "exists": path.is_file(),
                    "bytes": path.stat().st_size if path.is_file() else None,
                    "sha256": sha256(path) if path.is_file() else None,
                }
            attempts.append({
                "id": slot_id,
                "attempt": attempt.get("attempt"),
                "status": attempt.get("status"),
                "actual_success": (attempt.get("metrics") or {}).get("success"),
                "quality_errors": attempt.get("quality_errors") or [],
                "files": files,
            })
    return attempts


def verify_attempt_reference(
    root: Path,
    current: List[Mapping[str, Any]],
    expected: List[Mapping[str, Any]],
) -> Tuple[bool, List[str]]:
    issues = []
    expected_by_id = {(item["id"], item["attempt"]): item for item in expected}
    current_by_id = {(item["id"], item["attempt"]): item for item in current}
    if set(current_by_id) != set(expected_by_id):
        issues.append("attempt_set_changed")
    for identity, reference in expected_by_id.items():
        actual = current_by_id.get(identity)
        if actual is None:
            continue
        if actual.get("status") != reference.get("status"):
            issues.append(f"attempt_status_changed:{identity[0]}:{identity[1]}")
        if actual.get("actual_success") != reference.get("actual_success"):
            issues.append(f"attempt_outcome_changed:{identity[0]}:{identity[1]}")
        if actual.get("quality_errors") != reference.get("quality_errors"):
            issues.append(f"attempt_quality_changed:{identity[0]}:{identity[1]}")
        for role, expected_file in (reference.get("files") or {}).items():
            actual_file = (actual.get("files") or {}).get(role)
            if not actual_file or not actual_file.get("exists"):
                issues.append(f"attempt_file_missing:{identity[0]}:{identity[1]}:{role}")
            elif actual_file.get("sha256") != expected_file.get("sha256"):
                issues.append(f"attempt_file_hash_changed:{identity[0]}:{identity[1]}:{role}")
    return not issues, issues


def write_record_cards(
    batch: Path,
    records: List[Mapping[str, Any]],
    start_condition: Mapping[str, Any],
) -> None:
    for record in records:
        card = {
            "schema": "start-condition-record-card-v1",
            "record_id": record["id"],
            "source_session": record["source_session"],
            "start_condition": start_condition,
            "labels": record["labels"],
            "run_conditions": record["run_conditions"],
            "run_result": {
                "metrics": record["metrics"],
                "phase_history": (record.get("result") or {}).get("phase_history"),
                "timing": record["timing"],
            },
            "file_manifest": record["files"],
            "quality": record["quality"],
        }
        write_json(batch / "records" / record_filename(record["id"]), card)


def write_report(
    batch: Path,
    records: List[Mapping[str, Any]],
    completeness: Mapping[str, Any],
    mode: str,
    reference_ok: bool,
    start_condition: Mapping[str, Any],
    attempt_integrity_issues: List[str],
) -> Dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    failures = []
    for record in records:
        for category, issues in record["quality"]["issues"].items():
            issue_counts[category] += len(issues)
        if not record["quality"]["passed"]:
            failures.append({"id": record["id"], "issues": record["quality"]["issues"]})
    state = json.loads((batch / "batch-state.json").read_text(encoding="utf-8"))
    attempt_file_count = sum(
        len(attempt.get("files") or {}) + (1 if attempt.get("launcher_log") else 0)
        for slot in state.get("slots", {}).values()
        for attempt in slot.get("attempts", [])
    )
    unaccepted_slots = []
    for slot_id, slot in state.get("slots", {}).items():
        if slot.get("accepted_attempt"):
            continue
        unaccepted_slots.append({
            "id": slot_id,
            "attempts": [
                {
                    "attempt": attempt.get("attempt"),
                    "session": attempt.get("session"),
                    "actual_success": (attempt.get("metrics") or {}).get("success"),
                    "status": attempt.get("status"),
                    "quality_errors": attempt.get("quality_errors") or [],
                }
                for attempt in slot.get("attempts", [])
            ],
        })
    report = {
        "schema": "start-condition-quality-report-v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "start_condition": start_condition,
        "batch_counts": state.get("counts"),
        "record_count": len(records),
        "tracked_file_count": sum(len(record["files"]) for record in records),
        "tracked_attempt_file_count": attempt_file_count,
        "variant_completeness": completeness,
        "reference_manifest_checksum_valid": reference_ok,
        "attempt_integrity_issues": attempt_integrity_issues,
        "issue_counts": dict(issue_counts),
        "failed_records": failures,
        "unaccepted_slots": unaccepted_slots,
        "passed": len(records) == 30 and completeness["passed"] and reference_ok and not attempt_integrity_issues and not failures,
    }
    write_json(batch / "quality-report.json", report)
    counts = report["batch_counts"] or {}
    lines = [
        "# 新开始条件质量报告",
        "",
        f"- 开始条件：`{start_condition['id']}`；方块初始位置：`{start_condition['cube_initial_position_m']}`。",
        f"- 检查模式：`{mode}`；会话数：**{len(records)}/30**；受保护文件：**{report['tracked_file_count']}**。",
        f"- 全部尝试（含作废）受保护文件：**{report['tracked_attempt_file_count']}**。",
        f"- 成功结果：**{counts.get('successful_outcomes', 0)}**；预期失败结果：**{counts.get('failed_outcomes', 0)}**。",
        f"- 全部实际尝试中的成功：**{counts.get('attempt_successful_outcomes', 0)}**；失败：**{counts.get('attempt_failed_outcomes', 0)}**；无结果：**{counts.get('attempts_without_outcome', 0)}**。",
        f"- 实际尝试：**{counts.get('total_attempts', 0)}**；重跑：**{counts.get('rerun_attempts', 0)}**；作废：**{counts.get('void_attempts', 0)}**。",
        f"- 总体结果：**{'通过' if report['passed'] else '失败'}**。",
        "",
        "| 检查项 | 结果 |",
        "| --- | --- |",
        f"| 文件完整性 | {'通过' if not issue_counts['file_availability'] else '失败'} |",
        f"| 标签、运行条件与结果一致 | {'通过' if not issue_counts['label_consistency'] else '失败'} |",
        f"| 标签与视频日志绑定 | {'通过' if not issue_counts['label_video_consistency'] else '失败'} |",
        f"| 视频和运行时间对齐 | {'通过' if not issue_counts['timing'] else '失败'} |",
        f"| 30 个版本槽位完整 | {'通过' if completeness['passed'] else '失败'} |",
        f"| 文件哈希未改变 | {'通过' if not issue_counts['integrity'] and reference_ok and not attempt_integrity_issues else '失败'} |",
        "",
    ]
    if failures or unaccepted_slots or not completeness["passed"]:
        lines.extend(["## 问题", ""])
        for failure in failures:
            lines.append(f"- `{failure['id']}`: `{json.dumps(failure['issues'], ensure_ascii=False)}`")
        for slot in unaccepted_slots:
            lines.append(
                f"- 未接受槽位 `{slot['id']}`: "
                f"`{json.dumps(slot['attempts'], ensure_ascii=False)}`"
            )
        if completeness["missing_ids"]:
            lines.append(f"- 缺少有效版本：`{', '.join(completeness['missing_ids'])}`")
        for issue in attempt_integrity_issues:
            lines.append(f"- 尝试证据完整性：`{issue}`")
    else:
        lines.extend(["## 结论", "", "该小批次 30 条数据通过现有质量规则，实际尝试和作废记录均保留在 `batch-state.json`。"])
    (batch / "quality-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    batch = args.batch_dir if args.batch_dir.is_absolute() else root / args.batch_dir
    reference_path = batch / "reference-manifest.json"
    checksum_path = batch / "reference-manifest.sha256"
    mode = (
        "initialize"
        if args.initialize
        else "refresh"
        if args.refresh
        else "status"
        if args.status
        else "verify"
    )
    reference_ok = True
    attempt_integrity_issues: List[str] = []
    baseline: Dict[str, Mapping[str, Any]] = {}

    try:
        batch_state = json.loads((batch / "batch-state.json").read_text(encoding="utf-8"))
        start_condition = batch_state["start_condition"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Cannot load batch start condition: {exc}", file=sys.stderr)
        return 2

    try:
        selected = load_batch_selection(root, batch)
    except (OSError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Cannot load batch selection: {exc}", file=sys.stderr)
        return 2
    completeness = completeness_check(selected)

    if args.status:
        pass
    elif args.initialize:
        if reference_path.exists() or checksum_path.exists():
            print("Refusing to overwrite the batch reference; use --verify.", file=sys.stderr)
            return 2
    elif args.refresh:
        if not reference_path.is_file() or not checksum_path.is_file():
            print("Cannot refresh because the previous reference is missing.", file=sys.stderr)
            return 2
    else:
        if not reference_path.is_file() or not checksum_path.is_file():
            print("Batch reference manifest is missing; use --initialize once.", file=sys.stderr)
            return 2
        expected_hash = checksum_path.read_text(encoding="utf-8").split()[0]
        reference_ok = sha256(reference_path) == expected_hash
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        baseline = {record["id"]: record for record in reference.get("records", [])}
        current_attempts = attempt_reference(root, batch_state)
        attempts_ok, attempt_integrity_issues = verify_attempt_reference(
            root, current_attempts, reference.get("attempts", [])
        )
        reference_ok = reference_ok and attempts_ok

    records: List[Dict[str, Any]] = []
    for item in selected:
        snapshot = snapshot_record(root, item)
        record = validate_snapshot(root, item, snapshot, baseline.get(item["id"]))
        condition_errors = start_condition_issues(record, start_condition)
        if condition_errors:
            record["quality"]["issues"]["start_condition"] = condition_errors
            record["quality"]["checks"]["start_condition"] = False
            record["quality"]["passed"] = False
        records.append(record)

    write_record_cards(batch, records, start_condition)

    if args.refresh:
        preflight_passed = (
            len(records) == 30
            and completeness["passed"]
            and all(record["quality"]["passed"] for record in records)
        )
        if not preflight_passed:
            print("Reference was not refreshed because repaired data still fails quality checks.", file=sys.stderr)
            return 1
        archive = batch / "reference-history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive.mkdir(parents=True, exist_ok=False)
        shutil.copy2(reference_path, archive / reference_path.name)
        shutil.copy2(checksum_path, archive / checksum_path.name)
        report_path = batch / "quality-report.json"
        if report_path.is_file():
            shutil.copy2(report_path, archive / report_path.name)

    if args.initialize or args.refresh:
        reference = {
            "schema": "start-condition-reference-manifest-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "start_condition": start_condition,
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
            "attempts": attempt_reference(root, batch_state),
        }
        write_json(reference_path, reference)
        checksum_path.write_text(f"{sha256(reference_path)}  reference-manifest.json\n", encoding="utf-8")

    report = write_report(
        batch, records, completeness, mode, reference_ok, start_condition,
        attempt_integrity_issues,
    )
    print(f"Wrote start-condition quality report to {batch / 'quality-report.md'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
