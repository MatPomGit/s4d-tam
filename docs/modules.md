# S4D-TAM module catalog

This page maps the architecture described in the paper to concrete repository modules and separates executable reference components from target research modules that still require implementation or publication-quality validation.

## Executable reference modules

| Module | Role | Current maturity |
| --- | --- | --- |
| `token.py` | persistent 4D token state and attributes | executable reference |
| `proposal.py` | candidate token proposal | executable reference |
| `association.py` | feature, radial and fallback data association | executable reference |
| `memory.py` | token lifecycle, bounded history, resource budgets and memory updates | executable reference |
| `encoders/` | modality-oriented encoder interfaces, masking and fusion | executable interfaces; learned encoders still pending |
| `attention.py` | attention-related token scoring and state processing | executable reference; final learned hierarchy pending |
| `calibration.py` | calibration and uncertainty parameter support | executable utilities; held-out calibration pending |
| `reference_map.py` | reference-map representation and coordinate handling | executable reference |
| `topology.py` | topological graph and matching support | executable reference |
| `forecasting.py` | causal probabilistic occupancy and motion forecasting | executable reference |
| `planner.py` | deterministic risk, energy, time, progress and information-aware planning | executable reference |
| `telemetry.py` | versioned token-event and decision logging | executable reference |
| `pipeline.py` | integrated `S4DTAMReference` execution path | executable reference |

The current reference is intentionally deterministic and auditable where possible. This is valuable for numerical validation and ablation studies even when a later optimized learned implementation may achieve better task performance.

## Target research modules and extensions

The paper contains several mechanisms that are not yet fully represented by dedicated production-quality modules. They remain explicit roadmap items rather than empty code placeholders.

### Learned hierarchical spatiotemporal model

**Target responsibility:** learned multimodal embeddings, hierarchical local/global token attention and end-to-end temporal state updates.

**Current basis:** `encoders/`, `attention.py`, `memory.py`, `pipeline.py`.

**Completion criterion:** frozen training procedure, learned weights, held-out validation, numerical parity between research and optimized execution paths.

### Probabilistic scene-state inference

**Target responsibility:** Dynamic Bayesian Network or equivalent probabilistic state backend, multimodal future-state hypotheses and optional particle approximation for non-Gaussian scene uncertainty.

**Current basis:** `forecasting.py` provides causal probabilistic occupancy and flow distributions without a particle filter.

**Planned boundary:** a forecasting backend interface should allow the current causal forecaster and a future particle/learned backend to emit the same forecast contract.

### MPPI planning backend

**Target responsibility:** stochastic model-predictive trajectory optimization for higher-dimensional flight dynamics.

**Current basis:** `planner.py` uses deterministic beam search with explicit kinematic constraints and decomposed costs. This reference is preferred for initial confirmatory testing because identical inputs produce identical candidate evaluation and an auditable decision trace.

**Planned boundary:** add MPPI behind the same planner input/output contract and compare it against the deterministic reference before making it the default.

### Adaptive octree and level-of-detail world model

**Target responsibility:** hierarchical spatial resolution, LOD compression and risk/distance/resource-aware refinement.

**Current basis:** token lifecycle and `ResourceBudgets` already bound token count, memory and history, but they do not implement the full octree/LOD formulation from the paper.

**Planned boundary:** a world-model storage backend should preserve token identity and evaluator-visible semantics while changing only spatial storage/refinement policy.

### Active perception controller

**Target responsibility:** schedule RGB, thermal and other sensing/processing actions using expected information gain, uncertainty, energy and computational cost.

**Current basis:** planner information-value terms and modality masking provide the required signals, but no dedicated closed-loop sensor scheduler exists yet.

### Safety-aware real-time scheduler

**Target responsibility:** queue monitoring, deadline-aware resource allocation, graceful degradation and guaranteed minimum service for safety-critical stages.

**Current basis:** bounded token/resource budgets and telemetry expose some overload signals, but EDF-RT scheduling and predictive queue control are not implemented.

## Implementation policy

New optimized or learned modules should not bypass the normalized benchmark contracts. A replacement backend is acceptable only when it:

1. preserves causal data access;
2. emits the same evaluator-facing result schema;
3. records configuration and provenance;
4. has parity tests against the corresponding reference implementation where numerical equivalence is expected;
5. can be disabled independently when it corresponds to H1-H7.

See [Two-level comparison protocol](comparison-protocol.md) for how system-level and mechanism-level evaluation are kept separate.
