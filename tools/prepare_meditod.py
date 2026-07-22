"""MediTODを共通会話schemaへ前処理するCLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.meditod_dataset import (
    download_public_raw,
    file_sha256,
    load_yaml,
    prepare_canonical_full,
    prepare_public_raw,
    write_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="MediTOD前処理")
    parser.add_argument("--config", default="configs/datasets/meditod.yaml")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dialogs")
    parser.add_argument("--annotations")
    parser.add_argument("--source-mode", choices=("public_raw", "canonical_full"))
    parser.add_argument("--canonical-data-dir")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    mode = args.source_mode or str(config.get("source_mode", "public_raw"))
    seed = args.seed if args.seed is not None else int(config.get("split_seed", 42))
    output = Path(args.output_root)
    data_dir = output / "data"
    source_dir = output / "source"
    if mode == "canonical_full":
        if not args.canonical_data_dir:
            raise ValueError("canonical_fullには--canonical-data-dirが必要です。")
        conversations, samples, report = prepare_canonical_full(
            args.canonical_data_dir,
            config=config,
        )
        source_metadata = {
            "official_repository": config["official_repository"],
            "requested_revision": config["revision"],
            "canonical_data_dir": str(Path(args.canonical_data_dir)),
            "provided_by_user": True,
            "umls_licensed_data": True,
            "license": config.get("license", {}),
        }
        write_jsonl(conversations, data_dir / "meditod_conversations.jsonl")
        write_jsonl(samples, data_dir / "meditod_assistant_samples.jsonl")
        metadata = {
            "dataset": "MediTOD",
            "dataset_version": "meditod_canonical_full",
            "source_mode": mode,
            "source": source_metadata,
            "config": config,
            "statistics": report,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "preprocessing_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"MediTOD canonical前処理成果物を書き出しました: {output}")
        return 0
    if bool(args.dialogs) != bool(args.annotations):
        raise ValueError("--dialogsと--annotationsは両方指定してください。")
    if args.dialogs:
        dialogs = Path(args.dialogs)
        annotations = Path(args.annotations)
        source_metadata = {
            "official_repository": config["official_repository"],
            "requested_revision": config["revision"],
            "source_files": {
                dialogs.name: {"path": str(dialogs), "sha256": file_sha256(dialogs)},
                annotations.name: {"path": str(annotations), "sha256": file_sha256(annotations)},
            },
            "provided_by_user": True,
            "license": config.get("license", {}),
        }
    else:
        dialogs, annotations, source_metadata = download_public_raw(config, source_dir)
    conversations, samples, report = prepare_public_raw(
        dialogs,
        annotations,
        config=config,
        seed=seed,
    )
    write_jsonl(conversations, data_dir / "meditod_conversations.jsonl")
    write_jsonl(samples, data_dir / "meditod_assistant_samples.jsonl")
    metadata = {
        "dataset": "MediTOD",
        "dataset_version": config["dataset_version"],
        "source_mode": mode,
        "source": source_metadata,
        "config": config,
        "statistics": report,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"MediTOD前処理成果物を書き出しました: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
