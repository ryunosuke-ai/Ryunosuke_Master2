"""過去runから再利用可能なMathDial/WildChatデータだけを検証して複製する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from core.transition_bayes_model import load_transition_bayes_model
from tools.analyze_mathdial_corpus_transition_bayes import evaluate_emission_quality


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
BASIS_FILES = (
    "basis_model/mathdial_analysis_corpus.jsonl",
    "basis_model/mathdial_analysis_corpus.manifest.json",
    "basis_model/mathdial_analysis_input.txt",
    "basis_model/mathdial_analysis_prompt.txt",
    "basis_model/mathdial_model_quality.json",
    "basis_model/mathdial_transition_bayes_model.json",
    "basis_model/mathdial_transition_bayes_model.manifest.json",
    "basis_model/mathdial_transition_compat.json",
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
    stage = {
        "preprocess": "preprocess",
        "basis": "build_basis",
        "wildchat": "extract_wildchat",
    }[mode]
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


def validate_basis_artifacts(source_root: Path) -> dict[str, Any]:
    """再利用するbasisモデルのhash、schema、emission品質を再検証する。"""
    basis = source_root / "basis_model"
    model_path = basis / "mathdial_transition_compat.json"
    fine_model_path = basis / "mathdial_transition_bayes_model.json"
    quality = _load_json(basis / "mathdial_model_quality.json")
    model_manifest = _load_json(
        basis / "mathdial_transition_bayes_model.manifest.json"
    )
    analysis_manifest = _load_json(basis / "mathdial_analysis_corpus.manifest.json")
    if not quality.get("passed"):
        raise ValueError("再利用元のMathDial basis品質gateが不合格です。")
    if model_manifest.get("output_sha256") != sha256(fine_model_path):
        raise ValueError("再利用元のbasis model hashがmanifestと一致しません。")
    analysis_path = basis / "mathdial_analysis_corpus.jsonl"
    if analysis_manifest.get("output_sha256") != sha256(analysis_path):
        raise ValueError("再利用元のbasis分析標本hashがmanifestと一致しません。")
    conversation_path = source_root / "mathdial/data/mathdial_conversations.jsonl"
    if analysis_manifest.get("input_sha256") != sha256(conversation_path):
        raise ValueError("再利用元のbasis入力とMathDial正規化会話が一致しません。")
    load_transition_bayes_model(model_path)
    current_quality = evaluate_emission_quality(
        _load_json(model_path),
        margin=float(quality.get("required_margin", 0.10)),
        min_negative_observations=int(
            quality.get("minimum_negative_dominant_observations", 2)
        ),
    )
    if not current_quality["passed"]:
        raise ValueError("再利用元のbasisモデルが現在のemission品質gateに不合格です。")
    return current_quality


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
    if mode == "basis":
        basis_quality = validate_basis_artifacts(source_root)
        files = BASIS_FILES
    else:
        basis_quality = None
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
    if basis_quality is not None:
        manifest["modes"][mode]["emission_quality"] = basis_quality
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
    parser.add_argument(
        "--mode", choices=("preprocess", "basis", "wildchat"), required=True
    )
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
