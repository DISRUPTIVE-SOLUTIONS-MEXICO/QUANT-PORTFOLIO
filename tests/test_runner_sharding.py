"""Tests for the walk-forward batch sharding and partial-merge plumbing."""

from __future__ import annotations

import numpy as np
import pandas as pd

import run_xcdr_v3_parallel_research as runner


def test_slice_schedule_partitions_without_overlap():
    schedule = [{"i": i} for i in range(17)]
    shards = [runner.slice_schedule(schedule, k, 3) for k in range(3)]
    flat = [task["i"] for shard in shards for task in shard]
    assert sorted(flat) == list(range(17))
    # Contiguous chunks: each shard is an increasing run.
    for shard in shards:
        ids = [t["i"] for t in shard]
        assert ids == sorted(ids)


def test_slice_schedule_passthrough_and_bounds():
    schedule = [{"i": i} for i in range(5)]
    assert runner.slice_schedule(schedule, -1, 0) == schedule
    assert runner.slice_schedule(schedule, 0, 1) == schedule
    assert runner.slice_schedule(schedule, 9, 3) == []


def test_partial_artifacts_roundtrip(tmp_path):
    results = pd.DataFrame(
        {
            "test_start": pd.to_datetime(["2024-01-02", "2024-03-01"]),
            "objective": ["policy_a", "policy_b"],
            "active_ann_return": [0.05, 0.02],
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "objective": ["policy_a", "policy_a"],
            "active_return": [0.001, -0.002],
            "test_start": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        }
    )
    weights = pd.DataFrame(
        {
            "test_start": pd.to_datetime(["2024-01-02"]),
            "objective": ["policy_a"],
            "ticker": ["AAA"],
            "weight": [0.10],
        }
    )
    runner.write_partial_artifacts(tmp_path, 0, results, daily, weights)
    runner.write_partial_artifacts(tmp_path, 1, results.assign(objective="policy_c"), daily, weights)
    merged_results, merged_daily, merged_weights = runner.load_partial_artifacts(tmp_path)
    assert len(merged_results) == 4
    assert len(merged_daily) == 4
    assert len(merged_weights) == 2
    # Date-like columns are restored as datetimes for downstream pivots.
    assert np.issubdtype(merged_results["test_start"].dtype, np.datetime64)
    assert np.issubdtype(merged_daily["date"].dtype, np.datetime64)


def test_load_partial_artifacts_empty_dir(tmp_path):
    results, daily, weights = runner.load_partial_artifacts(tmp_path)
    assert results.empty and daily.empty and weights.empty


def test_batch_config_exposes_pit_universe_flag():
    cfg = runner.BatchConfig(pit_universe=True)
    assert cfg.pit_universe is True
    assert runner.BatchConfig().pit_universe is False
