# Publication path: S4D-TAM

## 1. Target outlet

**Primary journal:** Journal of Field Robotics (Wiley)  
**Planned category:** Regular Article  
**Publication discipline:** automatyka, elektronika, elektrotechnika i technologie kosmiczne (AEEiTK)  
**Polish ministerial score used for planning:** 140 points; re-check the binding ministerial list immediately before submission.

Journal of Field Robotics is the primary target because the manuscript combines a new autonomous-navigation representation with sensing, persistent world modelling, prediction and planning for aerial robots operating in GNSS-degraded and otherwise unstructured environments. The journal explicitly prioritizes robotics with both theoretical and practical significance and expects empirical validation beyond simulation-only evidence.

## 2. Submission model

The manuscript will be prepared as a **full Regular Article**, not as a conceptual paper, research note or simulation-only contribution. Submission is blocked until the central performance claims are supported by reproducible quantitative experiments and at least one representative field or field-analog validation.

The initial submission will use the journal's free-format route. The source of record is LaTeX. `main.tex` assembles independently maintained section files from `sections/`.

## 3. Experimental phases

The repository distinguishes two external-comparison phases explicitly:

- `study_phase: development` is used for vertical slices, integration tests and methodological debugging. It may intentionally contain a sensor-compatible subset of independent baselines and its results must not be promoted to confirmatory evidence;
- `study_phase: confirmatory` is used for the frozen publication study and requires the complete core baseline set defined by the repository protocol.

The first implemented development vertical slice is `configs/experiments/tartanair_orb_slam3_development.yaml`. It evaluates TartanAir with S4D-TAM and ORB-SLAM3 through the same evaluator after dataset-freeze and baseline-evidence gates. The operational procedure is documented in `docs/tartanair-orb-slam3-development.md`.

## 4. Required evidence before submission

### Gate A — frozen benchmark definition

Before confirmatory runs:

1. freeze the dataset × modality × baseline matrix;
2. record source versions, selected sequences and SHA-256 manifests;
3. freeze train/development/test roles;
4. freeze preprocessing and coordinate-frame conventions;
5. freeze primary and secondary metrics;
6. freeze exclusion and failure rules.

Target public datasets are TartanAir, Blackbird UAV Dataset, MARSIM and AeroVerse. A baseline is evaluated only where the required sensor modalities are genuinely available. Unsupported dataset–algorithm pairs must be reported as not applicable, not silently omitted.

### Gate B — external system comparison

The confirmatory comparison should include, where modality-compatible:

- ORB-SLAM3;
- VINS-Mono;
- FAST-LIO2;
- LIO-SAM;
- S4D-TAM reference implementation.

At minimum report trajectory accuracy and robustness measures appropriate to the modality, together with runtime, memory use and failure rate. Aggregate results must preserve per-sequence results rather than reporting only global means.

### Gate C — internal mechanism ablation

Run controlled ablations isolating the contribution of:

- semantic token representation;
- persistent temporal memory;
- uncertainty modelling;
- occupancy forecasting;
- hierarchical/spatiotemporal attention;
- risk/information-aware planning terms.

The ablation design must use the same frozen cohorts and metric implementation as the full model.

### Gate D — statistical analysis

For primary endpoints:

- report effect sizes and confidence intervals in addition to p-values;
- use paired comparisons when algorithms are evaluated on the same sequences;
- correct for multiplicity where multiple confirmatory comparisons are made;
- disclose failed runs and missing outputs;
- retain machine-readable per-run results.

### Gate E — field or field-analog validation

JFR requires experimental evidence that is not limited to simulation. Before submission, include at least one representative validation using a real UAV or a defensible field analog/HIL setup. It should exercise the claims most specific to S4D-TAM, especially operation under degraded positioning, persistent spatial memory and planning under partial observability.

The validation must report platform, sensors, compute hardware, environment, scenario protocol, number of runs, failure criteria and safety constraints.

## 5. Manuscript restructuring

The canonical article structure is:

1. Introduction
2. Literature Review
3. Research Gap and Contributions
4. S4D-TAM System Concept
5. Mathematical Description
6. Research Hypotheses
7. Experimental Validation
8. Discussion
9. Conclusions
10. Declarations

Each numbered section is stored as a separate `.tex` file. `main.tex` contains only document-level formatting, author metadata, abstract, keywords, section inclusion and bibliography commands.

## 6. JFR-specific preparation rules

The working manuscript follows the following submission constraints:

- single-column, readable 12 pt initial-submission layout;
- abstract no longer than 250 words;
- exactly seven keywords;
- SI units;
- sequentially numbered sections, equations, figures and tables;
- figures and tables embedded in the review manuscript with self-contained captions;
- consistent numbered citations;
- title page with complete author and affiliation information;
- data availability, funding, conflict-of-interest and ethics statements;
- ORCID supplied in the Wiley submission system;
- high-resolution final figure assets retained separately.

## 7. Bibliography audit

Before submission, audit every cited work against a primary publisher page, DOI registry or official dataset/project page. In particular, dataset references for TartanAir, Blackbird, MARSIM and AeroVerse must point to the correct dataset publications or canonical project records. No placeholder, guessed, mismatched or AI-invented citation may remain.

Acceptance criterion: every bibliography item used in the final manuscript has verified authorship, title, year, venue and DOI/URL where applicable.

## 8. Reproducibility package

Before submission:

1. tag the exact source revision used for the paper;
2. archive the tagged release with a persistent identifier (e.g. Zenodo DOI);
3. freeze benchmark manifests and configuration files;
4. export machine-readable per-run and aggregate results;
5. generate manuscript figures and tables directly from those frozen results;
6. include environment/container metadata sufficient to recreate the computational workflow.

## 9. Submission decision gate

The manuscript is **GO for submission** only when all conditions below are satisfied:

- no unsupported quantitative claims remain;
- all primary benchmark experiments are complete;
- ablation study is complete;
- field/field-analog validation is complete;
- statistical analysis is complete;
- all tables and figures are generated from frozen artifacts;
- bibliography audit has no unresolved entries;
- abstract and conclusions contain final quantitative findings rather than future-work language;
- declarations are complete;
- repository release and persistent archive are available;
- final PDF compiles without errors and has no unresolved references/citations.

Until then, the manuscript remains a research draft and must not be submitted as a full empirical JFR article.

## 10. Fallback outlets

If JFR becomes unsuitable after the empirical study is complete, reassess journals above the required ministerial threshold. IEEE Robotics and Automation Letters is a technically strong alternative but would require substantial compression and restructuring. The fallback decision must be based on the binding ministerial list and journal author guidelines valid on the actual submission date.
