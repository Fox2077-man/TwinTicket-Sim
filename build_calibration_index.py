"""Build and validate the canonical 30-session calibration index.

The three symptom experiments intentionally share one normal run per symptom.
Each symptom then contributes the three treatments (injected-fault,
wrong-cause, and true-fix) for three distinct single-cause interventions:
``1 + (3 * 3) = 10``.  This program does not create or alter simulations; it
only selects the approved session directories and produces an auditable index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


VARIANTS = ("injected-fault", "wrong-cause", "true-fix")
REQUIRED_FILES = ("experiment.log", "summary.json", "run-001/result.json", "run-001/trace.jsonl", "run-001.mp4")
CONTROL_FIELDS = ("seed", "device", "ik_method", "reset_settle_steps", "post_run_settle_steps", "task_contract")
LOG_ERROR = re.compile(r"\b(ERROR|TRACEBACK|FATAL|Exception)\b")

SYMPTOMS: Mapping[str, Dict[str, Any]] = {
    "cannot-grasp": {
        "label": "抓不到",
        "normal": "artifacts/fault1-comparison/normal",
        "causes": {
            "gripper-command-open": {
                "label": "抓夹未关闭",
                "variants": "artifacts/fault1-comparison",
            },
            "zero-cube-friction": {
                "label": "方块接触摩擦为零",
                "variants": "artifacts/fault2-comparison",
            },
            "high-grasp-approach": {
                "label": "抓取高度偏高",
                "variants": "artifacts/fault3-comparison",
            },
        },
    },
    "midway-drop": {
        "label": "中途掉落",
        "normal": "artifacts/midway-drop-comparison/normal",
        "causes": {
            "gripper-release": {"label": "运输中打开抓夹"},
            "friction-loss": {"label": "运输中禁用指块碰撞"},
            "motion-disturbance": {"label": "运输中向下速度扰动"},
        },
    },
    "inaccurate-placement": {
        "label": "抓住并移动，但放置不准",
        "normal": "artifacts/misplacement-comparison/normal",
        "causes": {
            "planner-x-offset": {"label": "规划释放目标 X 偏移"},
            "transport-y-bias": {"label": "运输控制命令 Y 偏差"},
            "early-release": {"label": "到目标前打开抓夹"},
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/calibration-30"))
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_session(directory: Path) -> Optional[Path]:
    sessions = [path for path in directory.glob("session-*") if path.is_dir()]
    return max(sessions, key=lambda path: path.stat().st_mtime) if sessions else None


def relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def session_record(
    root: Path,
    symptom: str,
    cause: str,
    variant: str,
    directory: Path,
) -> Dict[str, Any]:
    session = latest_session(directory)
    record: Dict[str, Any] = {
        "id": f"{symptom}/{cause}/{variant}",
        "symptom": symptom,
        "cause": cause,
        "variant": variant,
        "source_directory": relative(directory, root),
        "session": relative(session, root) if session else None,
    }
    if session is None:
        record.update({"evidence_complete": False, "validation_errors": ["missing_session"]})
        return record

    required = {name: session / name for name in REQUIRED_FILES}
    missing = [name for name, path in required.items() if not path.is_file()]
    summary = read_json(required["summary.json"]) if not missing or required["summary.json"].is_file() else {}
    result = read_json(required["run-001/result.json"]) if required["run-001/result.json"].is_file() else {}
    log = required["experiment.log"].read_text(encoding="utf-8") if required["experiment.log"].is_file() else ""
    record.update(
        {
            "evidence": {name: relative(path, root) if path.is_file() else None for name, path in required.items()},
            "sha256": {name: sha256(path) if path.is_file() else None for name, path in required.items()},
            "config": summary.get("config"),
            "metrics": result.get("metrics"),
            "phase_history": result.get("phase_history"),
            "simulation_steps": result.get("simulation_steps"),
            "evidence_complete": not missing,
            "experiment_log_clean": not bool(LOG_ERROR.search(log)),
            "validation_errors": [f"missing:{name}" for name in missing],
        }
    )
    if not record["experiment_log_clean"]:
        record["validation_errors"].append("experiment_log_contains_error_marker")
    return record


def source_directory(root: Path, symptom: str, cause: str, variant: str) -> Path:
    spec = SYMPTOMS[symptom]
    if variant == "normal":
        return root / spec["normal"]
    cause_spec = spec["causes"][cause]
    if "variants" in cause_spec:
        return root / cause_spec["variants"] / variant
    if symptom == "midway-drop":
        return root / "artifacts/midway-drop-comparison" / cause / variant
    return root / "artifacts/misplacement-comparison" / cause / variant


def all_records(root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for symptom, spec in SYMPTOMS.items():
        records.append(session_record(root, symptom, "shared-normal", "normal", source_directory(root, symptom, "shared-normal", "normal")))
        for cause in spec["causes"]:
            for variant in VARIANTS:
                records.append(session_record(root, symptom, cause, variant, source_directory(root, symptom, cause, variant)))
    return records


def metric(record: Mapping[str, Any], key: str) -> Any:
    return (record.get("metrics") or {}).get(key)


def equal_values(values: Iterable[Any]) -> bool:
    encoded = {json.dumps(value, sort_keys=True) for value in values}
    return len(encoded) == 1 and encoded != {"null"}


def symptom_checks(symptom: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    normal = next(record for record in records if record["variant"] == "normal")
    treatments = [record for record in records if record["variant"] != "normal"]
    control = {
        field: equal_values((record.get("config") or {}).get(field) for record in records)
        for field in CONTROL_FIELDS
    }
    errors: List[str] = []
    if len(records) != 10:
        errors.append(f"expected_10_records_got_{len(records)}")
    if not metric(normal, "success"):
        errors.append("normal_did_not_succeed")
    if symptom == "cannot-grasp":
        for record in treatments:
            successful = bool(metric(record, "success"))
            expected = record["variant"] == "true-fix"
            if successful != expected:
                errors.append(f"outcome_mismatch:{record['id']}")
    elif symptom == "midway-drop":
        for record in treatments:
            expected = record["variant"] == "true-fix"
            if bool(metric(record, "success")) != expected:
                errors.append(f"outcome_mismatch:{record['id']}")
            if expected and metric(record, "drop_detected"):
                errors.append(f"true_fix_still_dropped:{record['id']}")
            if not expected and not (metric(record, "grasp_established_before_fault") and metric(record, "drop_detected")):
                errors.append(f"drop_evidence_missing:{record['id']}")
    else:
        for record in treatments:
            expected = record["variant"] == "true-fix"
            if bool(metric(record, "success")) != expected:
                errors.append(f"outcome_mismatch:{record['id']}")
            if expected and metric(record, "misplacement_detected"):
                errors.append(f"true_fix_still_misplaced:{record['id']}")
            if not expected and not (
                metric(record, "lifted")
                and not metric(record, "drop_detected")
                and metric(record, "misplacement_detected")
            ):
                errors.append(f"misplacement_evidence_missing:{record['id']}")
    if not all(control.values()):
        errors.append("task_controls_not_identical")
    for record in records:
        errors.extend(f"{record['id']}:{error}" for error in record.get("validation_errors", []))
    return {
        "selected_session_count": len(records),
        "controls_equal": control,
        "outcomes_valid": not any(error.startswith(("normal_", "outcome_", "true_fix_", "drop_", "misplacement_")) for error in errors),
        "valid": not errors,
        "validation_errors": errors,
    }


def cause_label(symptom: str, cause: str) -> str:
    return "共享正常抓取" if cause == "shared-normal" else SYMPTOMS[symptom]["causes"][cause]["label"]


def write_report(root: Path, output: Path, records: List[Dict[str, Any]], checks: Dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    grouped = {symptom: [record for record in records if record["symptom"] == symptom] for symptom in SYMPTOMS}
    manifest = {
        "schema": "canonical-calibration-30-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "one shared normal plus three variants for each of three single-cause interventions",
        "expected_total": 30,
        "selected_total": len(records),
        "symptom_checks": checks,
        "sessions": records,
    }
    (output / "calibration-index.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# 三种症状的 30 条校准数据",
        "",
        "每种症状只选择 1 条共享正常抓取，并选择 3 个单一原因的 `injected-fault`、`wrong-cause`、`true-fix` 各 1 条，因此每组恰好为 `1 + 3 x 3 = 10` 条；总数为 `10 + 10 + 10 = 30` 条。",
        "",
    ]
    for symptom, spec in SYMPTOMS.items():
        check = checks[symptom]
        lines.extend(
            [
                f"## {spec['label']} (`{symptom}`)",
                "",
                "| 编号 | 原因 | 版本 | 成功 | 视频 | 结果 | 轨迹 | 实验日志 | 会话摘要 |",
                "| --- | --- | --- | ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for index, record in enumerate(grouped[symptom], 1):
            evidence = record.get("evidence") or {}
            lines.append(
                f"| {index} | {cause_label(symptom, record['cause'])} | `{record['variant']}` | "
                f"`{metric(record, 'success')}` | `{evidence.get('run-001.mp4') or '缺失'}` | "
                f"`{evidence.get('run-001/result.json') or '缺失'}` | `{evidence.get('run-001/trace.jsonl') or '缺失'}` | "
                f"`{evidence.get('experiment.log') or '缺失'}` | `{evidence.get('summary.json') or '缺失'}` |"
            )
        lines.extend(
            [
                "",
                f"- 已选会话数：**{check['selected_session_count']}/10**；控制条件一致：**{all(check['controls_equal'].values())}**；结果模式有效：**{check['outcomes_valid']}**；证据与日志完整：**{check['valid']}**。",
                "",
            ]
        )
    lines.extend(
        [
            "## 校验结论",
            "",
            f"- 已选总数：**{len(records)}/30**。",
            f"- 三组均通过：**{all(check['valid'] for check in checks.values())}**。",
            "- 每条记录的 SHA-256、会话目录、配置、指标和验证状态见 `calibration-index.json`。",
        ]
    )
    (output / "calibration-index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    records = all_records(root)
    checks = {
        symptom: symptom_checks(symptom, [record for record in records if record["symptom"] == symptom])
        for symptom in SYMPTOMS
    }
    write_report(root, output, records, checks)
    complete = len(records) == 30 and all(check["valid"] for check in checks.values())
    print(f"Wrote calibration index to {output / 'calibration-index.md'}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
