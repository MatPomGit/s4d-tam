from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from s4dtam_benchmark.algorithms.s4dtam.memory import LifecycleRules, TokenMemory
from s4dtam_benchmark.algorithms.s4dtam.telemetry import (
    EVENT_LOG_SCHEMA,
    EventLogConfig,
    InMemoryEventSink,
    JsonlEventLogger,
)


def test_jsonl_logger_writes_versioned_ordered_events(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(
        path,
        dataset="dataset",
        sequence="sequence",
        algorithm="algorithm",
    )
    logger.emit("frame_completed", 1.25, token_count=3)

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["log_started", "frame_completed"]
    assert [record["event_index"] for record in records] == [0, 1]
    assert all(record["schema"] == EVENT_LOG_SCHEMA for record in records)
    assert records[1]["sequence_time_s"] == 1.25


def test_logger_rejects_unsafe_paths_and_reserved_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        EventLogConfig(directory="../outside")
    logger = JsonlEventLogger(
        tmp_path / "events.jsonl", dataset="d", sequence="s", algorithm="a"
    )
    with pytest.raises(ValueError, match="reserved"):
        logger.emit("invalid", 0.0, event_index=99)


def test_memory_emits_lifecycle_and_deterministic_pruning_events() -> None:
    sink = InMemoryEventSink()
    memory = TokenMemory(
        association_mode="radial",
        association_radius_m=0.1,
        max_tokens=1,
        lifecycle=LifecycleRules(sleep_after_s=1.0, remove_after_s=10.0),
        event_sink=sink,
    )
    first = memory.update(np.zeros(3), 0.0)
    memory.update(np.array([5.0, 0.0, 0.0]), 2.0)

    state_events = [event for event in sink.events if event["event"] == "token_state_changed"]
    pruning_events = [event for event in sink.events if event["event"] == "token_pruned"]
    assert state_events[0]["token_id"] == first.token_id
    assert state_events[0]["new_state"] == "sleeping"
    assert len(pruning_events) == 1
    assert {
        "attention_score",
        "local_attention",
        "temporal_attention",
        "global_attention",
        "map_bytes_before",
    } <= pruning_events[0].keys()


def test_logging_can_omit_attention_components() -> None:
    sink = InMemoryEventSink()
    memory = TokenMemory(
        association_mode="radial",
        association_radius_m=0.1,
        max_tokens=1,
        event_sink=sink,
        log_attention_components=False,
    )
    memory.update(np.zeros(3), 0.0)
    memory.update(np.ones(3), 1.0)
    event = next(item for item in sink.events if item["event"] == "token_pruned")
    assert "attention_score" in event
    assert "local_attention" not in event
