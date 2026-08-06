"""Gold-only DPO比較実験の監査、応答生成、Oracle入力統合。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tools.run_oracle_evaluation_lora_pair import (
    generate_reply_with_adapter,
    load_lora_bundle,
)


CONFIG_VERSION = "gold_only_dpo500.v1"
GOLD_MODEL_NAME = "gold_only"
GOLD_ADAPTER_NAME = "gold_only"
MODEL_ALIASES = {
    "base": "base",
    "Base": "base",
    "basis": "basis",
    "BASiS": "basis",
    "bayes_dpo": "basis",
    "random": "random_dpo",
    "Random": "random_dpo",
    "random_dpo": "random_dpo",
    GOLD_MODEL_NAME: GOLD_MODEL_NAME,
}
FOUR_MODELS = ("base", GOLD_MODEL_NAME, "basis", "random_dpo")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} が不正なJSONです。") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} はobjectである必要があります。")
            rows.append(row)
    return rows


def write_jsonl_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("version") != CONFIG_VERSION:
        raise ValueError(f"Gold-only config versionが不正です: {path}")
    if not isinstance(config.get("datasets"), dict):
        raise ValueError("Gold-only configにdatasetsがありません。")
    return config


def dataset_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    try:
        value = config["datasets"][dataset]
    except KeyError as exc:
        raise ValueError(f"未対応datasetです: {dataset}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"dataset configがmappingではありません: {dataset}")
    return value


def preference_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(key) or "") for key in ("prompt", "chosen", "rejected"))


def record_source_key(row: dict[str, Any]) -> tuple[str, str, str]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return (
        str(row.get("source_dataset") or metadata.get("source_dataset") or ""),
        str(row.get("source_dialogue_id") or metadata.get("conversation_id") or ""),
        str(row.get("turn_index") or metadata.get("assistant_turn_index") or ""),
    )


def validate_gold_rows(
    gold_rows: list[dict[str, Any]],
    basis_rows: list[dict[str, Any]],
    *,
    dataset: str,
    expected: int,
) -> dict[str, Any]:
    if len(gold_rows) != expected:
        raise ValueError(f"{dataset} gold件数が不正です: {len(gold_rows)}/{expected}")
    missing_fields: list[int] = []
    non_train: list[str] = []
    preference_keys: list[tuple[str, str, str]] = []
    source_keys: list[tuple[str, str, str]] = []
    for index, row in enumerate(gold_rows, start=1):
        key = preference_key(row)
        if any(not value.strip() for value in key):
            missing_fields.append(index)
        preference_keys.append(key)
        source_keys.append(record_source_key(row))
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        split = str(metadata.get("source_split") or metadata.get("split") or "").lower()
        if split != "train":
            non_train.append(f"{index}:{split or 'missing'}")
        translated_hash = str(metadata.get("translated_prompt_hash") or "")
        rejected_hash = str(metadata.get("rejected_prompt_hash") or "")
        if translated_hash and rejected_hash and translated_hash != rejected_hash:
            raise ValueError(f"{dataset} goldでchosen/rejected context hashが不一致です: {index}")
    if missing_fields:
        raise ValueError(f"{dataset} goldに空のprompt/chosen/rejectedがあります: {missing_fields[:20]}")
    if non_train:
        raise ValueError(f"{dataset} goldにtrain以外が含まれます: {non_train[:20]}")
    if len(set(preference_keys)) != expected:
        raise ValueError(f"{dataset} goldにDPO完全重複があります。")
    nonempty_source_keys = [key for key in source_keys if any(key)]
    if len(nonempty_source_keys) != len(set(nonempty_source_keys)):
        raise ValueError(f"{dataset} goldにsource turn重複があります。")

    basis_counter = Counter(preference_key(row) for row in basis_rows)
    absent = [index for index, key in enumerate(preference_keys, start=1) if basis_counter[key] < 1]
    if absent:
        raise ValueError(f"{dataset} goldが既存BASiS armに含まれません: {absent[:20]}")
    return {
        "records": len(gold_rows),
        "basis_records": len(basis_rows),
        "basis_membership_verified": True,
        "train_only": True,
        "preference_duplicates": 0,
        "source_turn_duplicates": 0,
        "source_datasets": sorted({key[0] for key in source_keys if key[0]}),
    }


def audit_evaluation_leakage(
    gold_rows: list[dict[str, Any]], evaluation_rows: list[dict[str, Any]], *, dataset: str
) -> dict[str, Any]:
    """小コーパスtrainと既存評価会話のID重複を検査する。"""

    gold_conversations = {
        str(row.get("source_dialogue_id") or "").strip() for row in gold_rows
    } - {""}
    evaluation_conversations = {
        str(row.get("conversation_id") or "").strip() for row in evaluation_rows
    } - {""}
    overlap = sorted(gold_conversations & evaluation_conversations)
    if overlap:
        raise ValueError(f"{dataset} goldと評価testに会話ID重複があります: {overlap[:20]}")
    evaluation_splits = {
        str(row.get("split") or "").lower() for row in evaluation_rows if row.get("split")
    }
    if dataset in {"mathdial", "meditod"} and evaluation_splits - {"test", "ood"}:
        raise ValueError(
            f"{dataset}評価sourceにtest/OOD以外のsplitがあります: {sorted(evaluation_splits)}"
        )
    return {
        "gold_conversation_ids": len(gold_conversations),
        "evaluation_conversation_ids": len(evaluation_conversations),
        "conversation_id_overlap": 0,
        "evaluation_splits": sorted(evaluation_splits),
    }


def prepare_data(
    *,
    config_path: Path,
    dataset: str,
    output: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    ds = dataset_config(config, dataset)
    source = Path(ds["gold_source"])
    basis_source = Path(ds["basis_train_source"])
    if not source.is_file() or not basis_source.is_file():
        raise FileNotFoundError(f"Gold/BASiS sourceがありません: {source} / {basis_source}")
    expected = int(config["expected_gold_records"])
    gold_rows = read_jsonl(source)
    evaluation_rows = read_jsonl(Path(ds["evaluation_source"]))
    audit = validate_gold_rows(
        gold_rows,
        read_jsonl(basis_source),
        dataset=dataset,
        expected=expected,
    )
    audit["evaluation_leakage"] = audit_evaluation_leakage(
        gold_rows, evaluation_rows, dataset=dataset
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    source_hash = sha256_file(source)
    output_hash = sha256_file(output)
    if source_hash != output_hash:
        raise RuntimeError("Gold training copyのSHA-256がsourceと一致しません。")
    training = dict(config["training"])
    effective_batch = int(training["gradient_accumulation_steps"])
    payload = {
        "version": CONFIG_VERSION,
        "created_at": utc_now(),
        "dataset": dataset,
        "baseline": "gold_only_dpo",
        "interpretation": (
            "目的小コーパス応答をchosen、同一contextの合成応答をrejectedとした"
            "Gold-only DPOであり、生コーパスchosenだけのSFTではない。"
        ),
        "source": source.as_posix(),
        "source_sha256": source_hash,
        "basis_train_source": basis_source.as_posix(),
        "basis_train_sha256": sha256_file(basis_source),
        "output": output.as_posix(),
        "output_sha256": output_hash,
        "audit": audit,
        "training": training,
        "estimated_optimizer_steps": math.ceil(expected / effective_batch),
        "comparison_note": "BASiSと同じ1 epochのため、データ量に応じて更新回数も少ない。",
    }
    write_json(manifest_path, payload)
    return payload


def evaluation_id(row: dict[str, Any]) -> str:
    value = str(row.get("sample_id") or row.get("prompt_id") or "").strip()
    if not value:
        raise ValueError("評価sourceにsample_id/prompt_idがありません。")
    return value


def generation_seed(row: dict[str, Any], *, default_seed: int, index: int) -> int:
    value = row.get("generation_seed")
    if isinstance(value, int):
        return value
    generation = row.get("generation")
    if isinstance(generation, dict) and isinstance(generation.get("seed"), int):
        return int(generation["seed"]) + index
    return default_seed + index


def generate_gold_responses(
    *,
    config_path: Path,
    dataset: str,
    lora_path: Path,
    output: Path,
    base_model: str,
    seed: int,
    mock: bool,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    ds = dataset_config(config, dataset)
    source_path = Path(ds["evaluation_source"])
    source_rows = read_jsonl(source_path)
    expected = int(ds["expected_eval_records"])
    if len(source_rows) != expected:
        raise ValueError(f"{dataset}評価source件数が不正です: {len(source_rows)}/{expected}")
    source_ids = [evaluation_id(row) for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{dataset}評価sourceのsample IDが重複しています。")
    existing_rows = read_jsonl(output) if output.exists() else []
    existing = {evaluation_id(row): row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("Gold-only既存応答にsample ID重複があります。")
    unexpected = sorted(set(existing) - set(source_ids))
    if unexpected:
        raise ValueError(f"Gold-only既存応答に今回対象外のsampleがあります: {unexpected[:20]}")

    generation = ds["generation"]
    bundle = None
    if not mock:
        bundle = load_lora_bundle(
            base_model,
            adapters={GOLD_ADAPTER_NAME: lora_path.as_posix()},
            use_4bit=False,
        )
    output_rows = list(existing.values())
    for index, row in enumerate(source_rows):
        sample_id = evaluation_id(row)
        if sample_id in existing:
            continue
        model_prompt = str(row.get("model_prompt") or "")
        if not model_prompt.strip():
            raise ValueError(f"{dataset}:{sample_id} に保存済みmodel_promptがありません。")
        current_seed = generation_seed(row, default_seed=seed, index=index)
        if mock:
            response = {
                "esconv": "その不安が続いていて、とてもつらいのですね。",
                "mathdial": "まず、どの計算から確認するとよいか考えてみましょう。",
                "meditod": "その症状がいつ始まり、どう変化したか教えてください。",
            }[dataset]
        else:
            assert bundle is not None
            response = generate_reply_with_adapter(
                bundle,
                model_prompt,
                adapter_name=GOLD_ADAPTER_NAME,
                max_new_tokens=int(generation["max_new_tokens"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                seed=current_seed,
            )
        if not response.strip():
            raise RuntimeError(f"{dataset}:{sample_id} のGold-only応答が空です。")
        source_hash = sha256_text(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        output_rows.append(
            {
                "sample_id": sample_id,
                "prompt_id": row.get("prompt_id", sample_id),
                "conversation_id": row.get("conversation_id"),
                "model_name": GOLD_MODEL_NAME,
                "model_prompt": model_prompt,
                "model_prompt_sha256": sha256_text(model_prompt),
                "gold_only_response": response,
                "generation_seed": current_seed,
                "generation": {
                    **generation,
                    "base_model": base_model,
                    "lora_path": lora_path.as_posix(),
                    "adapter_name": GOLD_ADAPTER_NAME,
                    "use_4bit": False,
                },
                "source_evaluation_record_sha256": source_hash,
                "source_ood": bool(row.get("ood")),
            }
        )
        write_jsonl_atomic(output_rows, output)
        print(
            f"[gold_only_generate] {dataset} {len(output_rows)}/{expected} {sample_id}",
            flush=True,
        )
    if len(output_rows) != expected:
        raise RuntimeError(f"{dataset} Gold-only応答が不足しています: {len(output_rows)}/{expected}")
    sorted_rows = sorted(output_rows, key=lambda row: source_ids.index(evaluation_id(row)))
    if manifest_path is not None:
        prompt_hash = sha256_text(
            "\n".join(
                f"{evaluation_id(row)}\t{sha256_text(str(row['model_prompt']))}"
                for row in source_rows
            )
        )
        write_json(
            manifest_path,
            {
                "version": CONFIG_VERSION,
                "created_at": utc_now(),
                "dataset": dataset,
                "records": len(sorted_rows),
                "evaluation_source": source_path.as_posix(),
                "evaluation_source_sha256": sha256_file(source_path),
                "model_prompt_set_sha256": prompt_hash,
                "responses": output.as_posix(),
                "responses_sha256": sha256_file(output),
                "generation": {**generation, "base_model": base_model, "seed": seed},
                "mock": mock,
            },
        )
    return sorted_rows


def _template_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[evaluation_id(row)].append(row)
    output: dict[str, dict[str, Any]] = {}
    for sample_id, rows in grouped.items():
        signatures = {
            json.dumps(
                {"prompt": row.get("prompt"), "history": row.get("history")},
                ensure_ascii=False,
                sort_keys=True,
            )
            for row in rows
        }
        if len(signatures) != 1:
            raise ValueError(f"Oracle templateのprompt/historyがモデル間で不一致です: {sample_id}")
        output[sample_id] = rows[0]
    return output


def build_oracle_input(
    *,
    config_path: Path,
    dataset: str,
    responses_path: Path,
    output: Path,
    ood_output: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = load_config(config_path)
    ds = dataset_config(config, dataset)
    responses = read_jsonl(responses_path)
    expected = int(ds["expected_eval_records"])
    if len(responses) != expected:
        raise ValueError(f"Gold-only評価応答が不足しています: {len(responses)}/{expected}")
    templates = (
        _template_by_sample(Path(ds["oracle_template_source"]))
        if ds.get("oracle_template_source")
        else {}
    )
    source_by_id = {
        evaluation_id(row): row for row in read_jsonl(Path(ds["evaluation_source"]))
    }
    main_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    for response_row in responses:
        sample_id = evaluation_id(response_row)
        source = source_by_id.get(sample_id)
        if source is None:
            raise ValueError(f"評価sourceにsampleがありません: {sample_id}")
        if templates:
            template = templates.get(sample_id)
            if template is None:
                raise ValueError(f"Oracle templateにsampleがありません: {sample_id}")
            row = {
                "sample_id": sample_id,
                "model_name": GOLD_MODEL_NAME,
                "prompt": template["prompt"],
                "history": template.get("history", []),
                "response": response_row["gold_only_response"],
                "metadata": {
                    **(template.get("metadata") or {}),
                    "gold_only": True,
                    "model_prompt_sha256": response_row["model_prompt_sha256"],
                },
            }
            is_ood = bool((template.get("metadata") or {}).get("ood"))
        else:
            row = {
                "sample_id": sample_id,
                "model_name": GOLD_MODEL_NAME,
                "prompt": source.get("prompt", ""),
                "history": source.get("history", []),
                "response": response_row["gold_only_response"],
                "category": source.get("category", ""),
                "metadata": {
                    "gold_only": True,
                    "model_prompt_sha256": response_row["model_prompt_sha256"],
                },
            }
            is_ood = False
        if not str(row["prompt"]).strip() or not str(row["response"]).strip():
            raise ValueError(f"Oracle入力のprompt/responseが空です: {sample_id}")
        (ood_rows if is_ood else main_rows).append(row)
    write_jsonl_atomic(main_rows, output)
    if ood_output is not None:
        write_jsonl_atomic(ood_rows, ood_output)
    expected_main = int(ds.get("expected_main_records", expected))
    expected_ood = int(ds.get("expected_ood_records", 0))
    if len(main_rows) != expected_main or len(ood_rows) != expected_ood:
        raise RuntimeError(
            f"{dataset} Oracle入力件数が不正です: main={len(main_rows)}/{expected_main} "
            f"ood={len(ood_rows)}/{expected_ood}"
        )
    return main_rows, ood_rows


def normalize_model(value: str) -> str:
    return MODEL_ALIASES.get(value, value)


def oracle_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("judge_model"),
        row.get("oracle_prompt_version"),
        row.get("oracle_eval_category"),
        int(row.get("score_scale", 0)),
        int(row.get("score_min", 0)),
        int(row.get("score_max", 0)),
    )


def merge_oracle_raw(
    *,
    existing_path: Path,
    gold_path: Path,
    output: Path,
    manifest_path: Path,
    expected_samples: int,
) -> dict[str, Any]:
    existing_rows = read_jsonl(existing_path)
    gold_rows = read_jsonl(gold_path)
    existing_models = {normalize_model(str(row.get("model_name") or "")) for row in existing_rows}
    if existing_models != {"base", "basis", "random_dpo"}:
        raise ValueError(f"既存Oracle rawの3モデル構成が不正です: {sorted(existing_models)}")
    if {normalize_model(str(row.get("model_name") or "")) for row in gold_rows} != {GOLD_MODEL_NAME}:
        raise ValueError("新規Oracle rawにGold-only以外が混入しています。")
    signatures = {oracle_signature(row) for row in existing_rows + gold_rows}
    if len(signatures) != 1:
        raise ValueError(f"既存・Gold-only Oracle条件が一致しません: {sorted(signatures, key=str)}")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing_rows + gold_rows:
        sample_id = evaluation_id(row)
        model = normalize_model(str(row["model_name"]))
        key = (sample_id, model)
        if key in by_key:
            raise ValueError(f"Oracle rawに重複があります: {sample_id}/{model}")
        normalized = dict(row)
        normalized["model_name"] = model
        by_key[key] = normalized
    sample_models: dict[str, set[str]] = defaultdict(set)
    for sample_id, model in by_key:
        sample_models[sample_id].add(model)
    incomplete = {
        sample: sorted(set(FOUR_MODELS) - models)
        for sample, models in sample_models.items()
        if models != set(FOUR_MODELS)
    }
    if len(sample_models) != expected_samples or incomplete:
        raise ValueError(
            f"4モデルOracle coverageが不正です: samples={len(sample_models)}/{expected_samples} "
            f"incomplete={dict(list(incomplete.items())[:10])}"
        )
    for sample_id in sample_models:
        contexts = {
            json.dumps(
                {
                    "prompt": by_key[(sample_id, model)].get("prompt"),
                    "history": by_key[(sample_id, model)].get("history", []),
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for model in FOUR_MODELS
        }
        if len(contexts) != 1:
            raise ValueError(f"Oracle 4モデル間でprompt/historyが不一致です: {sample_id}")
    merged = [by_key[key] for key in sorted(by_key)]
    write_jsonl_atomic(merged, output)
    payload = {
        "version": CONFIG_VERSION,
        "created_at": utc_now(),
        "existing_raw": existing_path.as_posix(),
        "existing_raw_sha256": sha256_file(existing_path),
        "gold_raw": gold_path.as_posix(),
        "gold_raw_sha256": sha256_file(gold_path),
        "combined_raw": output.as_posix(),
        "combined_raw_sha256": sha256_file(output),
        "samples": len(sample_models),
        "records": len(merged),
        "models": list(FOUR_MODELS),
        "oracle_signature": list(signatures)[0],
        "temporal_limitation": "既存3モデルとGold-onlyのOracle採点日は異なる。",
    }
    write_json(manifest_path, payload)
    return payload


def resolve_judge_model(config_path: Path, dataset: str) -> str:
    ds = dataset_config(load_config(config_path), dataset)
    if ds.get("judge_model"):
        return str(ds["judge_model"])
    env_name = str(ds.get("judge_model_env") or "")
    return str(os.getenv(env_name) or ds.get("judge_model_default") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gold-only DPO比較実験support")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/gold_only_dpo500_v1.yaml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset", required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--manifest", type=Path, required=True)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--dataset", required=True)
    generate_parser.add_argument("--lora-path", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--base-model", default="Qwen/Qwen3.5-27B")
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--mock", action="store_true")
    generate_parser.add_argument("--manifest", type=Path)

    oracle_parser = subparsers.add_parser("build-oracle-input")
    oracle_parser.add_argument("--dataset", required=True)
    oracle_parser.add_argument("--responses", type=Path, required=True)
    oracle_parser.add_argument("--output", type=Path, required=True)
    oracle_parser.add_argument("--ood-output", type=Path)

    merge_parser = subparsers.add_parser("merge-raw")
    merge_parser.add_argument("--existing", type=Path, required=True)
    merge_parser.add_argument("--gold", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--manifest", type=Path, required=True)
    merge_parser.add_argument("--expected-samples", type=int, required=True)

    resolve_parser = subparsers.add_parser("resolve-judge-model")
    resolve_parser.add_argument("--dataset", required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_data(
            config_path=args.config,
            dataset=args.dataset,
            output=args.output,
            manifest_path=args.manifest,
        )
    elif args.command == "generate":
        generate_gold_responses(
            config_path=args.config,
            dataset=args.dataset,
            lora_path=args.lora_path,
            output=args.output,
            base_model=args.base_model,
            seed=args.seed,
            mock=args.mock,
            manifest_path=args.manifest,
        )
    elif args.command == "build-oracle-input":
        build_oracle_input(
            config_path=args.config,
            dataset=args.dataset,
            responses_path=args.responses,
            output=args.output,
            ood_output=args.ood_output,
        )
    elif args.command == "merge-raw":
        merge_oracle_raw(
            existing_path=args.existing,
            gold_path=args.gold,
            output=args.output,
            manifest_path=args.manifest,
            expected_samples=args.expected_samples,
        )
    elif args.command == "resolve-judge-model":
        print(resolve_judge_model(args.config, args.dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
