"""JSONLの再開処理で使う小さな共通ユーティリティ。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def read_jsonl_records(
    path: Path | str,
    *,
    missing_ok: bool = False,
    strict: bool = True,
    label: str | None = None,
) -> tuple[list[Any], int]:
    """JSONLを読み込む。strict=Falseでは壊れた行を警告してskipする。"""
    input_path = Path(path)
    label_text = label or str(input_path)
    if not input_path.exists():
        if missing_ok:
            return [], 0
        raise FileNotFoundError(f"{label_text}が見つかりません: {input_path}")

    records: list[Any] = []
    skipped = 0
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                message = f"{label_text}の{line_number}行目をJSONとして読めません: {exc}"
                if strict:
                    raise ValueError(message) from exc
                skipped += 1
                print(f"[WARN] skip invalid JSONL line: {message}", file=sys.stderr, flush=True)
    return records, skipped


def ensure_jsonl_append_boundary(path: Path | str) -> None:
    """追記前に、既存JSONLの末尾が改行で終わるようにする。"""
    output_path = Path(path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return
    with output_path.open("rb+") as file:
        file.seek(-1, 2)
        if file.read(1) != b"\n":
            file.write(b"\n")
