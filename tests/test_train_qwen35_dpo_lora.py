"""Qwen3.5 DPO LoRA 学習スクリプトの軽量テスト。"""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from tools.train_qwen35_dpo_lora import (
    PreferenceDatasetSplit,
    build_model_kwargs,
    build_training_args,
    disable_peft_bitsandbytes_dispatch,
    parse_max_memory,
    print_dry_run_summary,
    read_preference_records,
    resolve_device_map_mode,
    split_records,
    summarize_records,
    validate_model_device_placement,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """テスト用JSONLを書き込む。"""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_row(index: int = 1) -> dict[str, str]:
    """DPOレコードを作る。"""
    return {
        "prompt": f"以下の会話の次のAI返答を生成してください。\n\nこれまでの会話:\nUser: 話題{index}\n\nAI:",
        "chosen": f"いいですね、話題{index}についてもう少し聞かせてください。",
        "rejected": f"話題{index}は一般的に重要ですね。",
        "metadata": {"source_rank": index},
    }


class FakeTorch:
    """モデル読み込み設定テスト用のtorchスタブ。"""

    bfloat16 = "bf16"
    float16 = "fp16"

    class cuda:
        @staticmethod
        def current_device() -> int:
            return 0


class FakeBitsAndBytesConfig:
    """BitsAndBytesConfig呼び出しを保持するスタブ。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeDPOConfig:
    """DPOConfig呼び出しを保持するスタブ。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTensor:
    """device属性だけを持つテンソルスタブ。"""

    def __init__(self, device: str):
        self.device = device


class FakeModel:
    """device配置検証テスト用モデルスタブ。"""

    def __init__(self, *, device_map=None, parameters=None, buffers=None):
        self.hf_device_map = device_map
        self._parameters = parameters or [("model.weight", FakeTensor("cuda:0"))]
        self._buffers = buffers or []

    def named_parameters(self):
        return list(self._parameters)

    def named_buffers(self):
        return list(self._buffers)


def make_args(**overrides) -> Namespace:
    """学習引数スタブを作る。"""
    values = {
        "no_4bit": False,
        "device_map": None,
        "force_single_gpu": False,
        "max_memory": None,
        "output_dir": "out",
        "num_train_epochs": 1,
        "max_steps": -1,
        "learning_rate": 5e-6,
        "beta": 0.1,
        "max_length": 1024,
        "max_prompt_length": 768,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "logging_steps": 1,
        "save_steps": 25,
        "save_total_limit": 2,
        "warmup_ratio": 0.03,
        "max_grad_norm": 0.3,
        "seed": 42,
    }
    values.update(overrides)
    return Namespace(**values)


def test_read_preference_records_keeps_only_required_columns(tmp_path: Path):
    path = tmp_path / "dpo.jsonl"
    write_jsonl(path, [make_row(1)])

    records = read_preference_records(path)

    assert records == [
        {
            "prompt": make_row(1)["prompt"],
            "chosen": make_row(1)["chosen"],
            "rejected": make_row(1)["rejected"],
        }
    ]


def test_read_preference_records_rejects_missing_required_column(tmp_path: Path):
    path = tmp_path / "dpo.jsonl"
    row = make_row(1)
    row.pop("rejected")
    write_jsonl(path, [row])

    with pytest.raises(ValueError, match="rejected"):
        read_preference_records(path)


def test_read_preference_records_rejects_empty_dataset(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="有効なレコード"):
        read_preference_records(path)


def test_split_records_uses_eval_ratio_and_seed():
    records = [make_row(index) for index in range(10)]

    first = split_records(records, eval_ratio=0.2, seed=123)
    second = split_records(records, eval_ratio=0.2, seed=123)

    assert len(first.train) == 8
    assert len(first.eval) == 2
    assert first == second
    assert {row["prompt"] for row in first.train}.isdisjoint({row["prompt"] for row in first.eval})


def test_split_records_can_disable_eval():
    records = [make_row(index) for index in range(3)]

    split = split_records(records, eval_ratio=0.0, seed=123)

    assert len(split.train) == 3
    assert split.eval == []


def test_summarize_records_returns_max_lengths():
    records = [make_row(1), make_row(2)]

    summary = summarize_records(records)

    assert summary["count"] == 2
    assert summary["max_prompt_chars"] == max(len(row["prompt"]) for row in records)
    assert summary["max_chosen_chars"] == max(len(row["chosen"]) for row in records)
    assert summary["max_rejected_chars"] == max(len(row["rejected"]) for row in records)


def test_print_dry_run_summary_outputs_core_settings(capsys):
    args = Namespace(
        dataset="artifacts/datasets/noxij_dpo_preferences_ai_user.jsonl",
        model_id="Qwen/Qwen3.5-27B",
        output_dir="artifacts/training_runs/qwen35_dpo_lora",
        no_4bit=False,
        device_map=None,
        force_single_gpu=False,
        max_memory=None,
    )
    split = PreferenceDatasetSplit(train=[make_row(1)], eval=[])

    print_dry_run_summary(args, split)

    output = capsys.readouterr().out
    assert "DPO LoRA dry-run" in output
    assert "Qwen/Qwen3.5-27B" in output
    assert "train/eval: 1 / 0" in output
    assert "4bit: 有効" in output
    assert "device_map: auto" in output


def test_resolve_device_map_defaults_by_quantization():
    assert resolve_device_map_mode(make_args(no_4bit=False)) == "auto"
    assert resolve_device_map_mode(make_args(no_4bit=True)) == "single"
    assert resolve_device_map_mode(make_args(device_map="none")) == "none"
    assert resolve_device_map_mode(make_args(device_map="auto", force_single_gpu=True)) == "single"


def test_parse_max_memory_converts_device_keys():
    assert parse_max_memory("0=46GiB,cpu=0GiB") == {0: "46GiB", "cpu": "0GiB"}


def test_parse_max_memory_rejects_invalid_format():
    with pytest.raises(ValueError, match="max-memory"):
        parse_max_memory("46GiB")


def test_build_model_kwargs_uses_auto_4bit_by_default():
    args = make_args(no_4bit=False, max_memory="0=46GiB,cpu=0GiB")
    deps = {"torch": FakeTorch, "BitsAndBytesConfig": FakeBitsAndBytesConfig}

    kwargs = build_model_kwargs(args, deps, dtype=FakeTorch.bfloat16)

    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"] == {0: "46GiB", "cpu": "0GiB"}
    assert kwargs["quantization_config"].kwargs["load_in_4bit"] is True
    assert "torch_dtype" not in kwargs


def test_build_model_kwargs_uses_single_gpu_for_no_4bit():
    args = make_args(no_4bit=True)
    deps = {"torch": FakeTorch, "BitsAndBytesConfig": FakeBitsAndBytesConfig}

    kwargs = build_model_kwargs(args, deps, dtype=FakeTorch.bfloat16)

    assert kwargs["device_map"] == {"": 0}
    assert kwargs["torch_dtype"] == FakeTorch.bfloat16
    assert "quantization_config" not in kwargs


def test_build_training_args_passes_max_steps():
    args = make_args(no_4bit=False, max_steps=2)
    deps = {"torch": FakeTorch, "DPOConfig": FakeDPOConfig}

    config = build_training_args(args, deps, dtype=FakeTorch.bfloat16, has_eval=False)

    assert config.kwargs["max_steps"] == 2
    assert config.kwargs["optim"] == "paged_adamw_8bit"


def test_build_training_args_limits_retained_checkpoints():
    args = make_args(save_total_limit=2)
    deps = {"torch": FakeTorch, "DPOConfig": FakeDPOConfig}

    config = build_training_args(args, deps, dtype=FakeTorch.bfloat16, has_eval=False)

    assert config.kwargs["save_steps"] == 25
    assert config.kwargs["save_total_limit"] == 2


def test_validate_model_device_placement_rejects_hf_offload():
    model = FakeModel(device_map={"": 0, "lm_head": "cpu"})

    with pytest.raises(RuntimeError, match="offload"):
        validate_model_device_placement(model)


def test_validate_model_device_placement_rejects_meta_parameter():
    model = FakeModel(parameters=[("model.layers.0.weight", FakeTensor("meta"))])

    with pytest.raises(RuntimeError, match="parameter"):
        validate_model_device_placement(model)


def test_validate_model_device_placement_accepts_cuda_model():
    model = FakeModel(device_map={"": 0})

    validate_model_device_placement(model)


def test_disable_peft_bitsandbytes_dispatch_forces_detectors_false():
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.model as peft_lora_model
    except ImportError:
        pytest.skip("peft がインストールされていません")

    disable_peft_bitsandbytes_dispatch()

    assert peft_import_utils.is_bnb_available() is False
    assert peft_import_utils.is_bnb_4bit_available() is False
    assert peft_lora_model.is_bnb_available() is False
    assert peft_lora_model.is_bnb_4bit_available() is False
