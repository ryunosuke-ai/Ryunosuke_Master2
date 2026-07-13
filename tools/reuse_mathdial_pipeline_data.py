"""過去runから再利用可能なMathDial/WildChatデータだけを検証して複製する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


CONFIGS = (
    "configs/datasets/mathdial.yaml",
    "configs/datasets/wildchat_tutoring.yaml",
)
PREPROCESS_FILES = (
    "mathdial/data/manifest.json",
    "mathdial/data/mathdial_assistant_samples.jsonl",
    "mathdial/data/mathdial_conversations.jsonl",
    "mathdial/data/mathdial_qid_overlap_quarantine.jsonl",
    "mathdial/reports/preprocessing_report.md",
    "mathdial/reports/preprocessing_summary.json",
)
WILDCHAT_FILES = (
    "wildchat/general_tutoring_candidates.jsonl",
    "wildchat/general_tutoring_conversations.jsonl",
    "wildchat/math_tutoring_candidates.jsonl",
    "wildchat/math_tutoring_conversations.jsonl",
    "wildchat/statistics.json",
    "wildchat/manifest.json",
    "wildchat/stream_checkpoint.json",
)


def sha256(path: Path) -> str:
    """ファイル内容のSHA-256を返す。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_count(path: Path) -> int | None:
    """JSONLだけ非空行数を返す。"""
    if path.suffix != ".jsonl":
        return None
    with path.open(encoding="utf-8", errors="replace") as file:
        return sum(bool(line.strip()) for line in file)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"再利用に必要なJSONを読めません: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"再利用に必要なJSONがobjectではありません: {path}")
    return payload


def validate_source(
    source_root: Path,
    *,
    mode: str,
    seed: int,
    project_root: Path,
) -> dict[str, Any]:
    """source runの成功marker、seed、dataset configを検証する。"""
    metadata = _load_json(source_root / "run_metadata.json")
    stage = "preprocess" if mode == "preprocess" else "extract_wildchat"
    marker = _load_json(source_root / "stage_state" / f"{stage}_SUCCESS.json")
    if marker.get("stage") != stage:
        raise ValueError(f"source側SUCCESS markerが不正です: {stage}")
    if int(metadata.get("seed", -1)) != seed or int(marker.get("seed", -1)) != seed:
        raise ValueError("source runとv3のseedが一致しません。")
    if marker.get("experiment_fingerprint") != metadata.get("experiment_fingerprint"):
        raise ValueError("source runのSUCCESS markerとrun metadataが一致しません。")
    source_configs = metadata.get("configs", {})
    for relative in CONFIGS:
        current = sha256(project_root / relative)
        if source_configs.get(relative) != current:
            raise ValueError(f"source runとdataset config hashが一致しません: {relative}")
    return {"run_metadata": metadata, "stage_marker": marker}


def reuse_files(
    source_root: Path,
    target_root: Path,
    *,
    mode: str,
    seed: int,
    project_root: Path,
) -> dict[str, Any]:
    """allowlistの成果物だけをhash照合して複製する。"""
    validation = validate_source(
        source_root, mode=mode, seed=seed, project_root=project_root
    )
    files = PREPROCESS_FILES if mode == "preprocess" else WILDCHAT_FILES
    copied = []
    for relative in files:
        source = source_root / relative
        target = target_root / relative
        if not source.is_file():
            raise ValueError(f"再利用元ファイルが不足しています: {source}")
        source_hash = sha256(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != source_hash:
            raise ValueError(f"再利用先に内容の異なるファイルがあります: {target}")
        if not target.exists():
            shutil.copy2(source, target)
        if sha256(target) != source_hash:
            raise ValueError(f"コピー後のhashが一致しません: {target}")
        copied.append(
            {
                "path": relative,
                "sha256": source_hash,
                "jsonl_records": line_count(source),
            }
        )
    manifest_path = target_root / "reuse_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {
        "source_run": str(source_root),
        "target_run": str(target_root),
        "seed": seed,
        "modes": {},
    }
    if manifest.get("source_run") != str(source_root) or manifest.get("seed") != seed:
        raise ValueError("既存reuse manifestのsourceまたはseedが一致しません。")
    manifest["modes"][mode] = {
        "source_stage": validation["stage_marker"]["stage"],
        "source_experiment_fingerprint": validation["run_metadata"].get("experiment_fingerprint"),
        "files": copied,
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="MathDial v3データ再利用")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--mode", choices=("preprocess", "wildchat"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()
    reuse_files(
        Path(args.source_root).resolve(),
        Path(args.target_root).resolve(),
        mode=args.mode,
        seed=args.seed,
        project_root=Path(args.project_root).resolve(),
    )
    print(f"[reuse] {args.mode} data copied from {args.source_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
