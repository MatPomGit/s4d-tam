# S4D-TAM manuscript

This directory is the canonical publication workspace for the S4D-TAM article.

## Target journal

Primary target: **Journal of Field Robotics (Wiley)**, manuscript category: **Regular Article**.

The journal is a strong fit because S4D-TAM addresses sensing, world modelling, planning, learning and autonomous operation of aerial robots in unstructured and dynamically changing environments. The publication strategy assumes that the final paper will contain both theoretical development and empirical validation in representative field or field-analog conditions.

## Source layout

- `main.tex` — root LaTeX source used to compile the manuscript;
- `sections/` — one `.tex` file per manuscript section;
- `references.bib` — BibTeX database;
- `PUBLICATION_PATH.md` — publication strategy and decision gates;
- `SUBMISSION_CHECKLIST.md` — final readiness checklist.

The previous sources in `paper/latex/` remain as historical working material. New publication-facing edits should be made in `manuscript/`.

## Build

From this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

or, without `latexmk`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Generated PDFs and auxiliary LaTeX files must not be treated as the source of record.

## Reproducibility rule

Numerical values in final tables and figures must be generated from frozen experiment outputs. Do not manually transcribe benchmark results into the manuscript. Every final result set should be traceable to dataset manifests, configuration files, software revision and experiment artifacts.
