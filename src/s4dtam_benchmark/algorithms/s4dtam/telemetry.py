"""Structured event logging for the S4D-TAM reference implementation.

Logs use an append-only JSON Lines schema.  Each line is independently parseable,
which makes partially written experiment artifacts useful after an interrupted run.
No wall-clock timestamp is included by default: sequence time and an increasing event
number provide reproducible ordering without leaking machine-dependent timing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Protocol


EVENT_LOG_SCHEMA = "s4dtam-token-events/v1"


class EventSink(Protocol):
    """Structural interface accepted by ``TokenMemory`` for event collection."""

    def emit(self, event: str, sequence_time_s: float | None, **fields: Any) -> None:
        """Persist one structured event."""


@dataclass(frozen=True, slots=True)
class EventLogConfig:
    """Configuration of per-sequence structured event logs.

    Args:
        enabled: Whether the pipeline creates an event log.
        directory: Directory below the experiment output directory.
        flush_each_event: Flush and fsync each line for crash durability. Disabling
            fsync improves throughput while retaining line-by-line appends.
        include_attention_components: Include local, temporal and global scores in
            pruning decision events.
    """

    enabled: bool = True
    directory: str = "logs"
    flush_each_event: bool = False
    include_attention_components: bool = True

    def __post_init__(self) -> None:
        path = Path(self.directory)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("event log directory must be a non-empty relative path")


class JsonlEventLogger:
    """Thread-safe writer for versioned S4D-TAM JSONL events.

    Args:
        path: Destination JSONL file. Parent directories are created automatically.
        dataset: Dataset identifier copied into every event.
        sequence: Sequence identifier copied into every event.
        algorithm: Algorithm identifier copied into every event.
        flush_each_event: Request an ``fsync`` after each event for crash durability.

    Notes:
        Event writes open the file in append mode for each line. This avoids a long-lived
        descriptor and ensures the artifact is readable while an experiment is running.
    """

    def __init__(
        self,
        path: Path,
        *,
        dataset: str,
        sequence: str,
        algorithm: str,
        flush_each_event: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = {"dataset": dataset, "sequence": sequence, "algorithm": algorithm}
        self.flush_each_event = flush_each_event
        self._event_index = 0
        self._lock = Lock()
        # A run owns its file, so stale content from an earlier run is never mixed in.
        self.path.write_text("", encoding="utf-8")
        self.emit("log_started", None)

    def emit(self, event: str, sequence_time_s: float | None, **fields: Any) -> None:
        """Validate and append one canonical JSON event.

        Args:
            event: Stable snake-case event name.
            sequence_time_s: Sensor/sequence timestamp, or ``None`` for run metadata.
            **fields: JSON-serializable event-specific values.

        Raises:
            ValueError: If ``event`` is empty or a field shadows an envelope key.
            TypeError: If a field is not JSON serializable.
        """
        if (
            not event
            or not event.islower()
            or not event[0].isalpha()
            or not event.replace("_", "").isalnum()
        ):
            raise ValueError("event must be a non-empty snake-case identifier")
        reserved = {"schema", "event", "event_index", "sequence_time_s", *self.context}
        collision = reserved.intersection(fields)
        if collision:
            raise ValueError(f"event fields shadow reserved keys: {sorted(collision)}")
        with self._lock:
            record = {
                "schema": EVENT_LOG_SCHEMA,
                "event": event,
                "event_index": self._event_index,
                "sequence_time_s": sequence_time_s,
                **self.context,
                **fields,
            }
            encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
                stream.flush()
                if self.flush_each_event:
                    os.fsync(stream.fileno())
            self._event_index += 1


class InMemoryEventSink:
    """Small event sink for tests, notebooks, and embedding applications."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: str, sequence_time_s: float | None, **fields: Any) -> None:
        """Append an event dictionary to :attr:`events`."""
        self.events.append({"event": event, "sequence_time_s": sequence_time_s, **fields})
