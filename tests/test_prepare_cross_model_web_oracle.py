"""Gemini/Claude Web用Oracleパケット生成のテスト。"""

from pathlib import Path

from tools.prepare_cross_model_web_oracle import (
    DatasetSource,
    batch_records,
    blind_records,
)


def fixture_source() -> DatasetSource:
    return DatasetSource(name="fixture", path=Path("unused.jsonl"), categories=())


def fixture_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item_index in range(4):
        for model_name in ("base", "basis", "random_dpo"):
            rows.append(
                {
                    "item_id": f"item_{item_index}",
                    "model_name": model_name,
                    "prompt": f"prompt {item_index}",
                    "history": [{"role": "user", "text": "history"}],
                    "response": f"response {model_name}",
                }
            )
    return rows


def test_blind_records_are_reproducible_and_hide_model_name() -> None:
    waves_a, key_a = blind_records(fixture_source(), fixture_rows(), seed=42)
    waves_b, key_b = blind_records(fixture_source(), fixture_rows(), seed=42)

    assert waves_a == waves_b
    assert key_a == key_b
    assert len(waves_a) == 3
    assert all(len(wave) == 4 for wave in waves_a)
    assert all("model_name" not in row for wave in waves_a for row in wave)
    assert {row["model_name"] for row in key_a} == {"base", "basis", "random_dpo"}


def test_batches_do_not_mix_same_item_responses() -> None:
    waves, _ = blind_records(fixture_source(), fixture_rows(), seed=42)
    batches = batch_records(waves, max_records=2, max_chars=100_000)

    assert sum(len(batch) for batch in batches) == 12
    for batch in batches:
        item_ids = [row["item_id"] for row in batch]
        assert len(item_ids) == len(set(item_ids))

