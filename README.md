# AdaFed

Code for the AdaFed paper. The experimental protocol is documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Install

Requires Python 3.10–3.12.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m adafed doctor --strict    # environment check
python -m adafed test               # unit tests
python -m adafed smoke              # short end-to-end check
```

## Make the figures

Compiling the PDFs needs `pdflatex` with `pgfplots`; add `--no-compile` to export the tables only. IJCNN1 is
downloaded automatically into `data/raw/`. All commands are resumable.

All figures of the paper (k-means split):

```bash
python -m adafed rates --split kmeans --jobs 8
python -m adafed benchmarks --split kmeans --jobs 8
python -m adafed figures --split kmeans
```

Only Figures 1 and 2:

```bash
python -m adafed figure rates-ijcnn-b1 --split kmeans        # Figure 1
python -m adafed figure selected-ijcnn1-32 --split kmeans    # Figure 2
```

The PDFs are written to `figures/rates/` and `figures/benchmarks/` and end in `-kmeans`.
