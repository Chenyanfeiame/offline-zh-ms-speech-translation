#!/usr/bin/env python3
"""Run reproducible HY-MT cross-request isolation tests.

This intentionally exercises the real local llama.cpp model.  Each numbered
test gets a newly started server process; requests inside a test share that
process but never share a messages array or session identifier.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from zhms_s2s.backends.hy_mt import HyMtLlamaTranslator
from zhms_s2s.config import Settings


CASES = {
    "A": {
        "text": "订单编号 A7392，请不要重复付款。",
        "source": "zh",
        "target": "ms",
    },
    "B": {
        "text": "请于 25 November 抵达 KL Sentral。",
        "source": "zh",
        "target": "ms",
    },
    "C": {
        "text": (
            "Jika sambungan Internet terputus, sistem masih akan menjalankan "
            "pengecaman, terjemahan dan sintesis suara secara setempat."
        ),
        "source": "ms",
        "target": "zh",
    },
}

C_FORBIDDEN_OUTPUT = (
    "KL Sentral",
    "吉隆坡",
    "A7392",
    "订单号",
    "订单编号",
    "重复付款",
    "请勿重复",
)


def build_backend(settings: Settings) -> HyMtLlamaTranslator:
    return HyMtLlamaTranslator(
        settings.mt_hy_model,
        settings.mt_llama_server_path,
        port=settings.mt_hy_port,
        gpu_layers=settings.mt_hy_gpu_layers,
        terminology_enabled=False,
        critical_slot_protection_enabled=False,
    )


def execute_case(
    backend: HyMtLlamaTranslator,
    test_name: str,
    case_name: str,
) -> dict:
    case = CASES[case_name]
    request_id = uuid.uuid4().hex
    started = time.perf_counter()
    translated, pivot, trace = backend.translate_with_trace(
        case["text"],
        case["source"],
        case["target"],
        request_id=request_id,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test": test_name,
        "case": case_name,
        "request_id": request_id,
        "raw_asr_text": case["text"],
        "actual_mt_input": case["text"],
        "actual_mt_request_messages": trace["actual_mt_request_messages"],
        "mt_raw_output": trace["mt_raw_output"],
        "final_translation": translated,
        "pivot_text": pivot,
        "model_id": trace["model_id"],
        "session_id": trace["session_id"],
        "mt_request_id": trace["mt_request_id"],
        "cache_prompt": trace["cache_prompt"],
        "latency_ms": elapsed_ms,
    }


def validate_record(record: dict) -> list[str]:
    errors: list[str] = []
    expected = CASES[record["case"]]["text"]
    if record["actual_mt_input"] != expected:
        errors.append("actual_mt_input differs from this case")
    messages = record["actual_mt_request_messages"]
    if len(messages) != 1 or messages[0].get("role") != "user":
        errors.append("request does not contain exactly one user message")
    else:
        content = str(messages[0].get("content", ""))
        if expected not in content:
            errors.append("current input is missing from the MT message")
        for other_name, other in CASES.items():
            if other_name != record["case"] and other["text"] in content:
                errors.append(f"message contains Case {other_name} input")
    if record["session_id"] is not None:
        errors.append("session_id is not null")
    if record["mt_request_id"] != record["request_id"]:
        errors.append("MT request_id does not match pipeline request_id")
    if record["cache_prompt"] is not False:
        errors.append("cache_prompt is not false")
    if record["case"] == "C":
        output = record["final_translation"]
        for forbidden in C_FORBIDDEN_OUTPUT:
            if forbidden.casefold() in output.casefold():
                errors.append(f"Case C output contains forbidden content: {forbidden}")
    return errors


def run_sequence(
    settings: Settings,
    test_name: str,
    case_names: tuple[str, ...],
    *,
    concurrent: bool = False,
) -> dict:
    backend = build_backend(settings)
    try:
        if concurrent:
            records: list[dict] = []
            with ThreadPoolExecutor(max_workers=len(case_names)) as pool:
                futures = {
                    pool.submit(execute_case, backend, test_name, name): name
                    for name in case_names
                }
                for future in as_completed(futures):
                    records.append(future.result())
            records.sort(key=lambda item: item["case"])
        else:
            records = [
                execute_case(backend, test_name, case_name)
                for case_name in case_names
            ]
    finally:
        backend.close()

    errors: list[str] = []
    for record in records:
        record_errors = validate_record(record)
        record["isolation_checks"] = {
            "passed": not record_errors,
            "errors": record_errors,
        }
        errors.extend(f"Case {record['case']}: {error}" for error in record_errors)
    return {
        "name": test_name,
        "sequence": list(case_names),
        "concurrent": concurrent,
        "passed": not errors,
        "errors": errors,
        "records": records,
    }


def prompt_payloads(settings: Settings) -> dict:
    old = HyMtLlamaTranslator(
        settings.mt_hy_model,
        settings.mt_llama_server_path,
        terminology_enabled=True,
        critical_slot_protection_enabled=True,
    )
    current = build_backend(settings)
    case = CASES["C"]
    old_payload = old._request_payload(
        old._prompt(case["text"], case["source"], case["target"])
    )
    # The previous implementation did not explicitly send this field.
    old_payload.pop("cache_prompt", None)
    current_payload = current._request_payload(
        current._prompt(case["text"], case["source"], case["target"])
    )
    return {
        "before": {
            "provenance": "reconstructed exactly from the backed-up pre-fix prompt/payload code",
            "payload": old_payload,
        },
        "after": {
            "provenance": "current production payload builder",
            "payload": current_payload,
        },
    }


def write_reports(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mt_isolation_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "payload_before_after.json").write_text(
        json.dumps(result["payload_before_after"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for test in result["tests"]:
        path = output_dir / f"{test['name'].lower().replace(' ', '_')}_audit.jsonl"
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in test["records"]
            ),
            encoding="utf-8",
        )

    lines = [
        "# MT 跨请求隔离实测",
        "",
        f"- 时间：{result['generated_at']}",
        f"- 总结论：{'PASS' if result['passed'] else 'FAIL'}",
        f"- model_id：{result['model_id']}",
        "- session_id：null",
        "- cache_prompt：false",
        "",
        "| Test | Sequence | Mode | Result | Case C translation |",
        "|---|---|---|---|---|",
    ]
    for test in result["tests"]:
        c_records = [record for record in test["records"] if record["case"] == "C"]
        c_output = c_records[0]["final_translation"] if c_records else "-"
        lines.append(
            f"| {test['name']} | {' → '.join(test['sequence'])} | "
            f"{'concurrent' if test['concurrent'] else 'sequential'} | "
            f"{'PASS' if test['passed'] else 'FAIL'} | {c_output.replace('|', '\\|')} |"
        )
    lines.extend(
        [
            "",
            "每次请求的完整 `request_id` / `actual_mt_input` / "
            "`actual_mt_request_messages` / `mt_raw_output` / `session_id` "
            "保存在同目录的 JSON/JSONL 审计文件中。",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--output-dir", default="reports/mt_request_isolation"
    )
    args = parser.parse_args()
    settings = Settings.load(args.config)
    if not settings.baseline_direct_pipeline:
        raise SystemExit("baseline_direct_pipeline must be true for this audit")
    if settings.mt_terminology_enabled or settings.mt_critical_slot_protection_enabled:
        raise SystemExit("all MT text interventions must be disabled for this audit")

    tests = [
        run_sequence(settings, "Test 1", ("C",)),
        run_sequence(settings, "Test 2", ("A", "C")),
        run_sequence(settings, "Test 3", ("B", "C")),
        run_sequence(settings, "Test 4", ("A", "B", "C")),
        run_sequence(settings, "Test 5", ("A", "B", "C"), concurrent=True),
    ]
    ids = [
        record["request_id"]
        for test in tests
        for record in test["records"]
    ]
    unique_ids = len(ids) == len(set(ids))
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": all(test["passed"] for test in tests) and unique_ids,
        "request_ids_unique": unique_ids,
        "model_id": "hy-mt1.5-1.8b",
        "session_id": None,
        "payload_before_after": prompt_payloads(settings),
        "tests": tests,
    }
    write_reports(Path(args.output_dir), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
