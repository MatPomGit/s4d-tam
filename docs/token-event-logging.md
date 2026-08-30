# S4D-TAM token event logging

## Purpose and guarantees

The reference pipeline writes a structured audit trail for every S4D-TAM sequence. The log is
intended for lifecycle debugging, resource-budget analysis, failure reconstruction, and
reproducible benchmark review. It is not a replacement for aggregate metrics in
`AlgorithmResult`.

The format identifier is `s4dtam-token-events/v1`. Logs are JSON Lines (`.jsonl`): every line is
a complete JSON object and can be processed independently. Events use sensor/sequence time,
not wall-clock time. `event_index` is a zero-based, strictly increasing order within one file.
Together these choices make lifecycle and pruning events reproducible even when measured
latency differs between hosts.

Raw images, point clouds, positions, descriptors, and embeddings are deliberately omitted.
The log contains identifiers, counts, scores, decisions, and timings. This keeps files small
and avoids duplicating potentially sensitive sensor data.

## Configuration

Configure logging in an S4D-TAM algorithm entry:

```yaml
event_logging:
  enabled: true
  directory: logs
  flush_each_event: false
  include_attention_components: true
  include_map_events: true
```

- `enabled` controls creation of the per-sequence artifact.
- `directory` is a relative path below the experiment output directory. Absolute paths and
  parent traversal (`..`) are rejected.
- `flush_each_event` calls `fsync` after every line. Enable it for crash-critical hardware
  trials; leave it disabled for normal offline benchmarks to avoid storage latency.
- `include_attention_components` adds local, temporal, and global scores to `token_pruned`.
  The aggregate score is always recorded.
- `include_map_events` records map-mode initialization, place-match decisions, tracking loss,
  and relocalization. Disable it when only token lifecycle telemetry is required. Raw map
  positions and descriptors are never written.

Files are stored as
`<output>/<directory>/<safe-dataset>/<safe-sequence>_s4d_tam_reference.jsonl`. Dataset and
sequence names are sanitized and receive a short SHA-256 suffix, preventing both traversal
and collisions between identifiers with the same sanitized spelling. The relative file name is also
reported as `metadata.event_log` in the algorithm result.

## Exact processing sequence

1. `S4DTAMReference.run` creates one `JsonlEventLogger` for a dataset sequence. The writer
   truncates a stale file and emits `log_started`.
2. `TokenMemory` receives the logger through the `EventSink` interface and emits
   `memory_initialized` with effective hard limits.
3. For each observation batch, memory emits `frame_started`, advances inactivity state, then
   runs joint association.
4. Association emits `token_matched`, `token_created`, or `proposal_discarded`. State changes
   are emitted separately, so activation, sleep, and reactivation remain directly observable.
5. Duplicate consolidation emits `tokens_merged`. The event identifies the stable survivor
   and removed identifier.
6. Hierarchical attention is evaluated before capacity enforcement. Each eviction emits
   `token_pruned`, including the deterministic aggregate score, map size before removal and,
   when configured, all attention components.
7. Update latency is measured after deterministic map operations. A breached latency SLO emits
   `time_budget_exceeded`; latency never participates in the pruning key.
8. `frame_completed` records counts, resident payload bytes, update latency and the SLO flag.
9. After the last sample, `run_completed` records final token, byte and sample counts.

When reference-map matching is enabled, `map_mode_initialized` records the map schema, graph
size and matching thresholds. Each available frame then produces either `map_match_accepted`
or `map_match_rejected`. Accepted events contain only token IDs, scalar scores, residual and
correction norm; rejected events aggregate reason codes without copying descriptors or poses.
`tracking_lost` records the transition into a lost state, while an accepted event sets
`relocalized=true` when it restores tracking.

Lifecycle-only time progression through `TokenMemory.advance` produces state/removal/pruning
events but no artificial frame event.

## Common envelope and event catalogue

Every event contains:

| Field | Meaning |
|---|---|
| `schema` | Stable format identifier. |
| `event` | Snake-case event type. |
| `event_index` | Total order within the file. |
| `sequence_time_s` | Sensor time or `null` for initialization. |
| `dataset`, `sequence`, `algorithm` | Run identity. |

Event-specific fields are additive. Consumers should reject an unknown major schema version,
but ignore unknown fields and event names from a compatible version.

| Event | Important fields |
|---|---|
| `log_started` | Common envelope only. |
| `memory_initialized` | Association mode and effective capacity limits. |
| `frame_started` | Candidate and resident-token counts. |
| `token_created` | Token ID, initial state, proposal confidence. |
| `token_matched` | Token/candidate IDs, confidence, association evidence. |
| `proposal_discarded` | Candidate ID, confidence and admission threshold. |
| `token_state_changed` | Token ID, previous/new state and reason. |
| `token_removed` | Token ID, state, inactivity and reason. |
| `tokens_merged` | Survivor ID, removed ID and resulting hit count. |
| `token_pruned` | Token ID, attention, capacity snapshot and reason. |
| `time_budget_exceeded` | Measured and configured milliseconds. |
| `frame_completed` | Association/output counts, bytes, latency and SLO status. |
| `map_mode_initialized` | Map mode/schema, place and transition counts, matching thresholds. |
| `map_match_accepted` | Token ID, candidate counts, confidence, residual, correction norm, relocalization flag. |
| `map_match_rejected` | Candidate/rejection counts, reason codes and tracking-loss transition. |
| `tracking_lost` | Sample index, availability state and loss reason. |
| `run_completed` | Final token/byte/sample counts and aggregate map-match/relocalization counts. |

## Reading and validating logs

This minimal reader preserves streaming behavior:

```python
import json
from pathlib import Path

path = Path("outputs/run/logs/synthetic-<hash>/sequence-<hash>_s4d_tam_reference.jsonl")
with path.open(encoding="utf-8") as stream:
    events = [json.loads(line) for line in stream]

assert all(event["schema"] == "s4dtam-token-events/v1" for event in events)
assert [event["event_index"] for event in events] == list(range(len(events)))
pruned_ids = [event["token_id"] for event in events if event["event"] == "token_pruned"]
```

JSON serialization rejects NaN and infinity. Envelope fields cannot be shadowed by custom
fields. The writer serializes keys canonically, protects event indices with a process-local
lock, appends one line per file open, flushes the Python stream, and optionally synchronizes
the underlying file. For multiple processes, use distinct per-sequence paths as the pipeline
does; a single file is not a cross-process aggregation target.

## Operational recommendations

- Use default buffered durability for development and offline replay.
- Enable `flush_each_event` for flight tests where power loss recovery is more important than
  latency fidelity.
- Archive JSONL files with the experiment YAML, run manifest, stdout/stderr and failures.
- Compare deterministic decision fields across reruns; compare latency statistically rather
  than byte-for-byte.
- Do not add raw sensor payloads to events. Store restricted inputs under the dataset's access
  policy and correlate them using sequence time.
