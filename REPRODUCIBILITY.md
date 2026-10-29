# Reproducibility specification

The executable specification is the source in `adafed/`. This document records the experiment design and the conventions needed to interpret the saved trajectories.

## Problems and data

Every problem is binary logistic regression without an intercept. Client `i` holds examples `(a_ij, y_ij)`, with `y_ij` in `{-1,+1}`, and

```text
ell(x; a,y) = log(1 + exp(-y a^T x))
F_i(x)      = mean_j ell(x; a_ij,y_ij) + r(x)
F(x)        = sum_i (n_i/n) F_i(x) = mean over all n rows of ell + r(x),     n = sum_i n_i.
```

So the objective is the mean over all training rows, and it is the same function for every client split of the same rows. In the paper's own splits all clients have `n_i = n/N` and `F` is the mean over clients. Every method's server aggregation weights client `i` by `n_i/n` (`1/N` for equal clients) and keeps its participation correction (AdaFed's Horvitz--Thompson scaling is unchanged); local losses and local steps are those of the client's own mean loss `F_i`. The evaluation, the reference values `F_star` and the smoothness `L` used for the baselines' client step sizes are those of the pooled rows (AdaFed's eta grid is centred differently; see the rate study below).

The three regimes use `lambda=0.01` and respectively

```text
smooth convex:    r(x) = (lambda/2) ||x||_2^2
nonsmooth convex: r(x) = lambda ||x||_1
smooth nonconvex: r(x) = lambda sum_j x_j^2/(1+x_j^2).
```

Convex performance is `F(x)-F_star`. Nonconvex performance is `||grad F(x)||_2^2`. Convex reference values were computed centrally in float64 by damped Newton for L2 and FISTA plus active-support Newton polishing for L1. The values and KKT metadata used by the paper are frozen in `data/fstar.json`; the solver implementations remain in `adafed/problem.py`.

| Dataset | Clients `N` | Examples/client `n_i` | Total | Dimension `d` | Partition block |
|---|---:|---:|---:|---:|---:|
| synthetic kappa 1 | 32 | 64 | 2,048 | 32 | 32 |
| synthetic kappa 10 | 32 | 64 | 2,048 | 32 | 32 |
| synthetic kappa 100 | 32 | 64 | 2,048 | 32 | 32 |
| IJCNN1 | 32 | 1,000 | 32,000 | 22 | 50 |

Synthetic data contain no pseudorandom draws. Row `i` and feature `j` start from the standard-normal quantile of the van der Corput value for index `i` in the `j`-th prime base. After column centering and reduced QR decomposition, column scales form a geometric sequence whose squared ratio gives pooled Gram condition number `kappa`; the largest eigenvalue of `A^T A/n` is fixed at one. The planted vector has nonzeros `2`, `-1`, and `0.5` at zero-based indices 0, 10, and 21. Labels compare the logistic probability to a separate van der Corput sequence in base 137.

Synthetic rows are stably sorted by planted margin, divided into blocks of 32, and dealt round-robin to clients. IJCNN1 uses the first 32,000 rows of LIBSVM's `ijcnn1.bz2`, without feature normalization. The required raw-file SHA-256 is

```text
450dee8470b168e47b9fae1fc160396b8a809c675737837fce3acd51dcdc0481
```

IJCNN1 rows are stably sorted by the margin of a centralized L1-logistic reference solution at `lambda=0.01`, divided into blocks of 50, and dealt round-robin. The data builder saves original row IDs in every generated `.npz` file.

**The shipped default partition.** The sort key is a floating-point margin `a_i . w_ref` (`exact_key` is false), so another CPU or BLAS can break a near-tie differently and build a different split. The default split is therefore shipped, like the k-means split: `data/ijcnn1-32-partition.json` holds the 32 x 1,000 row indices (clients in order, the rows of a client in the order they are dealt), the settings of the construction, the build's two reference numbers (`lambda_max` and the reference solve's KKT residual) and the SHA-256 of the partition as int64 values, row-major, `0d6c70541218a2e22e1b996c3b5d33828b6496dec0fd5b9831b101819c92bb5c`. `generate.build('ijcnn1-32')` loads this file and checks its shape, that it holds every one of the 32,000 rows once, and its digest; it does not solve the reference problem or sort, and a damaged file is refused. The digest is locked in `adafed/protocol.py` (`datasets.ijcnn1-32.partition_sha256`), so `protocol.validate()` fails on a different partition. The file was produced by the construction as it was before it was shipped; it equals the partition built on a second machine with a different CPU, and the one stored in the `ijcnn1-32.npz` that every earlier default-split result used. `python -m tools.default_partition` reruns the construction on this machine and prints `equal` or `DIFFERENT` against the tracked file (exit status 0 or 1); `--write` replaces the file with the regeneration. A regeneration that differs does not change the data or the figures, which use the tracked partition only. The Dirichlet splits (their draws are asserted by digest in `tests/test_dirichlet_small_alpha.py`) and the k-means partition are unchanged.

## AdaFed rate study

All runs start from `x_1=0` and use float64. Except for the explicit participation ablation, all 32 clients participate. The fixed parameter `eta` and accumulator initialization `S_1=gamma` give shared round step `eta_t=eta/S_t`. Local scaling is

```text
tau_(t,k) = 1                  for K=1
tau_(t,k) = 1/((K-1)t)        for K>1.
```

Rate panels use `K in {1,10,100}`. Exact mode averages all local examples at each local step. Stochastic mode draws fresh client-local minibatches uniformly with replacement at every round and local step. Primary stochastic panels use `b=1`; appendix panels use `b=8` on synthetic data and `b=32` on IJCNN1.

Synthetic rate runs use 2,000 communication rounds. Standard IJCNN1 runs use 1,000 rounds; the primary IJCNN1 `b=1` K-comparison uses a dedicated 10,000-round run. Eta/gamma and participation ablations use `K=10` and the standard dataset horizon. Checkpoints are round 1, every tenth round, and the final round.

Eta is tuned at `K=10`, `gamma=1`, and seed 1901, over 300 synthetic or 150 IJCNN1 rounds. A five-point integer power-of-ten grid is centered at `round(log10(1/L))`, with `L` the largest client smoothness plus the penalty's smoothness bound (`rate_study.local_smoothness`); the exact computed `1/L` is never used. If the selected point is on an edge, the grid extends by a decade, at most nine times, until the selected point is interior. Eta is held fixed across K. Synthetic tuning occurs at kappa 10 and is shared across synthetic condition numbers. Batch sizes are tuned separately. Gamma ablation uses `{0.01,0.1,1,10,100}` while fixing tuned eta. Participation uses `p in {0.25,0.5,1}` with independent Bernoulli client sampling and Horvitz--Thompson scaling.

## Adaptive-method benchmarks

Benchmarks use synthetic kappa 1 with stochastic minibatch `b=16` and IJCNN1 with stochastic minibatch `b=32`, `K=10`, full participation, and 10,000 communication rounds. The compared groups are:

- server adaptation: AdaFed, FedAdagrad, FedAdam, FedUR;
- client adaptation: AdaFed, Local AdaAlter, FedSPS, FedLADA;
- both levels: AdaFed, FAFED, FedAda2;
- selected comparison: AdaFed, FedAdagrad, FedLADA, FAFED.

AdaFed is jointly tuned over eta and gamma. Every baseline uses the method-specific power grid and finite search bounds in `rate_compare_selected.py` or `rate_compare_levels_10k.py`; edge selections extend by a decade. Tuning uses seed 1901 and 300 synthetic or 150 IJCNN1 rounds. Final runs use the selected settings without retuning at 10,000 rounds. The four selected-method trajectories are referenced in place by the grouped comparison, not copied.

**Selection rules as implemented.** The criterion is the final uniform-random-output mean of the tuning run (the objective gap, or the squared gradient norm for the nonconvex regime), and a run that diverged or did not finish never wins. The rule that picks among the finished candidates differs by method:

| Methods | Where | Rule |
|---|---|---|
| AdaFed (eta, gamma), in the selected comparison and the Figure 2 splits | `rate_compare_joint._best` | Every candidate within 0.1% of the minimum criterion (`criterion <= minimum + 0.001 max(abs(minimum), 1e-16)`) is kept, and among them the one with the smallest `abs(log10 eta) + abs(log10 gamma)` wins; ties go to the smaller criterion, then the smaller eta, then the smaller gamma. |
| FedAdagrad, FedLADA, FAFED (the Figure 2 baselines) | `rate_compare_selected.tune` | The plain minimum of the criterion; an exact tie goes to the candidate evaluated first. No tolerance and no preference for round settings. |
| FedAdam, Local AdaAlter, FedSPS, FedAda2, FedUR (the adaptation-level benchmark only) | `rate_compare_levels_10k._best` | The AdaFed rule: within 0.1% of the minimum, then the smallest `abs(log10 a) + abs(log10 b)` over the two tuned parameters, then the smaller criterion, then the smaller `a`, then `b`. |

So the Figure 2 baselines can be picked by a difference of less than 0.1% that AdaFed's rule would have resolved towards round settings; this asymmetry is part of the specification as run, not corrected.

**Grid edges that equal their search bounds.** A pick on a grid edge is extended by a decade only while the new value stays inside the declared bounds. For some baselines an initial grid edge already equals a bound, so that edge cannot be extended and a pick there is not bracketed (the selected-comparison tuning records keep an `interior` flag, false in that case):

| Method | Parameter | Initial grid | Bounds | Edges equal to a bound |
|---|---|---|---|---|
| FedAdagrad | server eta | 0.001 .. 10 | 0.001 .. 100 | lower edge 0.001 |
| FedLADA | client scale | 1e-5 .. 1e-1 | 1e-5 .. 10 | lower edge 1e-5 |
| FedLADA | server eta | 0.1 .. 10 | 0.01 .. 10 | upper edge 10 |
| FAFED | client scale | 1e-5 .. 1e-1 | 1e-5 .. 0.1 | both edges |
| FAFED | server eta | fixed at 1 | 1 .. 1 | not tuned |
| FedUR | alpha | 0.1, 0.5, 0.9, 1 | 0.1 .. 1 | both edges |

The grids and bounds of AdaFed, FedAdam, Local AdaAlter, FedSPS and FedAda2, and the other FedAdagrad parameter (client scale 1/128 .. 2 within 1/128000 .. 20), have room on both sides. FAFED's server eta is a fixed value, not a search. Because a pick on one of these edges was not bracketed, its tuned value is a boundary value, not an interior optimum.

## Randomness and reported outputs

Exact full-participation runs use seed 2027 and are deterministic. Rate-study stochastic curves use the twenty seeds 2027--2046, and so do the benchmark stochastic curves. Minibatch sampling uses a separate PyTorch generator seeded by `seed XOR 0x5DEECE66D`.

Metrics are evaluated on the pre-update server iterate `x_t`. A last-iterate curve reports the metric at `x_t`. The random-output curve is the exact running average of metric values over `x_1,...,x_t`, which equals the conditional expectation for an index drawn uniformly from those iterates. It is not the metric of an averaged iterate.

Across stochastic seeds, all plots, CSV summaries, and PGFPlots exports use the pointwise arithmetic mean. Every k-means figure, and Figures 1 and 2 (the IJCNN1 `b=1` rate panel and the selected IJCNN1 comparison) of every split, additionally shade asymmetric directional RMS-deviation bands (opacity 0.22 for the uniform random mean, 0.17 for the last iterate; tables thinned to 121 log-spaced rounds): deviations below and above the pointwise mean are squared and averaged separately. This construction respects the nonnegative metric domain without an arbitrary symmetric clipping rule. A floor of `1e-16` is applied only to values written for log-scale display; raw trajectory metrics are not clipped.

## Execution

Pinned Python dependencies are declared in both `pyproject.toml` and `requirements.txt`. The pipeline is resumable, validates saved configurations before reuse, and checks its scientific constants against `adafed/protocol.py`. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m adafed doctor --strict
python -m adafed test
python -m adafed smoke
python -m adafed all
```

The paper's figures are those of the k-means split. `python -m adafed all` runs the steps below for the default and the k-means splits, then Figures 1 and 2 for the Dirichlet splits, exports every table and compiles every figure. The steps can also be run one at a time with `--split S` (`default`, `kmeans`, `dirichlet0.01`, `dirichlet0.1`, `dirichlet0.5`, `dirichlet1`): `default` and `kmeans` run every step on that split's datasets, and a Dirichlet split restricts `rates` and `benchmarks` to its Figure 1 and Figure 2. The paper's figures alone:

```bash
python -m adafed rates --split kmeans --jobs N
python -m adafed benchmarks --split kmeans --jobs N
python -m adafed figures --split kmeans
```

The commands:

| Command | Does |
|---|---|
| `python -m adafed prepare` | builds the datasets (downloads IJCNN1 into `data/raw/`) and checks the convex references |
| `python -m adafed rates --jobs N` | AdaFed rates (`b=1/8/32`), eta/gamma ablations, partial participation, the 10,000-round IJCNN1 `b=1` panel |
| `python -m adafed benchmarks --jobs N` | tunes and runs the selected and adaptation-level comparisons |
| `python -m adafed export` | validates every required trajectory and writes the PGFPlots tables |
| `python -m adafed figures` | exports and compiles the standalone figures (needs `pdflatex` and `pgfplots`) |
| `python -m adafed figure NAME --split S` | one figure (`rates-ijcnn-b1` or `selected-ijcnn1-32`) of one split, from data to PDF |

`--jobs N` sets the worker processes (default: the usable CPUs, at most 8), `--no-compile` stops after the tables, and `--force-tune` (with `rates` or `all`) recomputes the rate-study tuning. `python run.py ...` accepts the same commands. A saved run is reused only when its manifest is complete and its configuration matches the requested run; only scheduled checkpoints are written, never model trajectories.

Outputs: figure sources and PDFs in `figures/rates/` and `figures/benchmarks/` (a split's versions end in `-kmeans`, `-dirichlet<alpha>`), their tables in each folder's `experiments-data/`, and tuning records and runs in `rate-study/` (Figure 1 runs in `rate-study/results/ijcnn-t10000/<dataset>/`, Figure 2 runs in `rate-study/comparison-selected-joint-10000/results/<dataset>/`).

Compute: Figure 1 or Figure 2 of one split takes 1.7--5.5 core-hours (runs and tuning); all twelve Figure 1/2 versions take 38.9 core-hours, and the synthetic and ablation studies of each full split come on top. The runs need only a CPU, and wall time scales as `1/N` with `--jobs N` down to the longest single run (a `K=100` exact run, 5--8 minutes on one core).

## IJCNN1 under Dirichlet label skew

Figures 1 (the 10,000-round IJCNN1 `b=1` rate panel) and 2 (the selected comparison) are repeated on four further client splits of the same 32,000 IJCNN1 rows into the same 32 clients, with Dirichlet label skew of concentration `alpha` in `{0.01, 0.1, 0.5, 1}` (Hsu et al., 2019). The split is selected by `--split dirichlet<alpha>`; the datasets are named `ijcnn1-32-dirichlet<alpha>`.

**Split.** For each class (`y=-1`, `y=+1`) its rows are shuffled and cut across the 32 clients at the cumulative proportions `q ~ Dirichlet(alpha 1_32)`; a draw that leaves any client without rows is discarded and redrawn from the same generator (NumPy `default_rng`, seed 2021, a protocol constant `generate.SPLIT_SEED`), at most 1,000 times. If none of the 1,001 draws leaves every client non-empty, the first draw is repaired: while a client is empty (lowest index first), one example, chosen by the same generator (which continues from where the last draw left it), moves to it from the currently largest client (lowest index among equals); `generate.dirichlet_split_info` returns the details. Alpha 0.1 needs 86 draws and alphas 0.5 and 1 one, so the cap and the repair never touch them (their splits are asserted by digest in `tests/test_dirichlet_small_alpha.py`); alpha 0.01 needs the repair. Rows within a client keep their original order. Client sizes are therefore unequal (alpha 0.01: 1 to 28,816 rows with 31 single-class clients, 26 of the 32 clients holding a single example the repair moved; alpha 0.1: 1 to 9,851 rows with 20 clients holding a single class; alpha 0.5: 5 to 7,810 rows, 1 single-class client; alpha 1: 35 to 3,611 rows, none). The data are stored flat with client offsets.

**Objective.** The objective is the one above, `F(x) = sum_i (n_i/n) F_i(x) + r(x)`, the mean over all `n = 32,000` rows, for every split (the builder asserts that every split holds the same rows). It therefore does not depend on `alpha`: `F_star`, the smoothness constant `L` used by the baselines' client rates, and the evaluated `F` and gradient norm are those of the pooled rows (each row weighs `1/n`). What the split changes is the clients: the server weights client `i` by `n_i/n` instead of `1/N` in AdaFed, FedAdagrad, FedLADA and FAFED (for the last three, the average over clients of the local models, and for FAFED also of its moment estimates; AdaFed keeps its `1/p` participation scaling), and the eta grid is centred on the largest client smoothness, as before. A client's stochastic minibatch is drawn uniformly with replacement from its own rows; exact local gradients use all of them. Unequal clients take a separate code path (`adafed/fl.py`, `adafed/adaptive_baselines.py`), tested against numpy references of every method and against the equal-client path on equal clients (`tests/test_ragged.py`); the default IJCNN1 path is untouched and gives byte-identical results.

**Protocol.** Everything else is the protocol above, per alpha and from scratch: tuning at `K=10` with seed 1901 (Figure 1: eta on the integer power-of-ten grid centred on the largest client smoothness, gamma 1, 150 rounds, stochastic `b=1` tuned apart from exact; Figure 2: the joint eta and gamma grid for AdaFed and the method-specific grids for the baselines, `b=32`), each grid extended while the pick is on an edge and inside the declared bounds, then 10,000-round runs with full participation, exact seed 2027 once and stochastic seeds 2027--2046. The figure sources `rates-ijcnn-b1-dirichlet<alpha>.tex` and `selected-ijcnn1-32-dirichlet<alpha>.tex` are the default figures' sources with their table names changed (`tools/figure_sources.py`; checked by a test), so styles, limits and the metadata include are identical.

**Reference values.** Because the objective is the same, `F_star` of every Dirichlet split equals the default split's (L2 0.4154526395889211, L1 0.42788364502097165) up to the solvers' last digit. The values of the eight Dirichlet entries are locked in `adafed/protocol.py` (and checked against `data/fstar.json`); running the pipeline from a fresh clone recomputes them and reproduces them exactly, and a test checks that they equal the default split's within 1e-12:

| alpha | F_star (L2, `lambda=0.01`) | F_star (L1, `lambda=0.01`) |
|---|---|---|
| 0.01 | 0.4154526395889211 | 0.4278836450209717 |
| 0.1 | 0.41545263958892104 | 0.4278836450209717 |
| 0.5 | 0.41545263958892104 | 0.4278836450209717 |
| 1 | 0.41545263958892115 | 0.4278836450209717 |
| default | 0.4154526395889211 | 0.42788364502097165 |

**Alpha 0.01 and 0.05.** Alpha 0.01 is used in the figures. Its split is built with the repair (26 examples moved, all from the largest client) and one client holds 28,816 of the 32,000 rows (about 90%); the other 31 clients hold 3,184 rows between them, 26 of them a single example. Alpha 0.05 (`ijcnn1-32-dirichlet0.05`, also built with the repair, 9 examples moved) is measured but not used in the figures: its `zeta^2` at the minimisers is within about 1% of alpha 0.1's (1.1% and 0.9%; 2.6% at `x = 0`).

**Floating point.** Runs are float64 and use no randomness except the seeds above; on a machine with a different BLAS the last digits of a trajectory can differ from the published ones.

## The k-means split

The paper's figures use a fifth client split of the same 32,000 IJCNN1 rows into the same 32 clients, built to make the clients' gradients as different as a balanced grouping can: `--split kmeans`, dataset `ijcnn1-32-kmeans` (the synthetic datasets' k-means splits are in the next section). Unlike the Dirichlet splits every client holds exactly 1,000 rows, so the data are stored like the default split's, the server weights are `1/N = n_i/n`, and the equal-client code path is the one that runs. The objective is the same pooled mean over all 32,000 rows, so `F_star` equals the default split's (locked in `adafed/protocol.py` and `data/fstar.json`: L2 0.4154526395889211, L1 0.4278836450209717).

**Reference point.** `x*` is the minimiser of the smooth convex (L2, `lambda = 0.01`) problem on the pooled rows, from the library's own solver (`problem.solve_l2sq`); it is the same for every split.

**Per-example gradients.** `g_j = sigmoid(b_j . x*) b_j` with `b_j = -y_j a_j`, the gradient at `x*` of row `j`'s logistic data loss (the penalty's gradient is the same for every row and cancels), for the 32,000 rows. For a split into clients, `zeta^2 = sum_i (n_i/n) ||grad l_i(x*) - grad l(x*)||^2` is the spread of the client means of the `g_j` around their overall mean: the part of the total spread `(1/n) sum_j ||g_j - g_bar||^2` that lies between the clients (the rest is inside them). A k-means grouping of the `g_j` minimises the inside part, so it maximises `zeta^2`.

**Balanced k-means.** `generate.balanced_kmeans` groups the `g_j` into 32 clusters of exactly 1,000 rows. It starts with k-means++ (the first centre uniform, each next one drawn with probability proportional to the squared distance to the nearest centre so far), gives each row, in a random order, the nearest centre with room left, and then alternates (a) an exact balanced assignment step and (b) recomputing the centres as the cluster means, until the assignment no longer changes. Step (a) is the transportation problem of sending every row to one client, every client receiving exactly 1,000 rows, at least total squared distance to the centres. `generate.balanced_assignment` solves it exactly by cancelling negative cycles of clients from the current assignment, and certifies the result: with `d` the final shortest-path distances on the 32 clients, every row's client minimises its cost minus `d` of the client (up to 1e-12 of the largest cost), which by linear-programming duality means that no balanced assignment is cheaper. A test compares the step with brute force on all balanced assignments of small instances. The construction uses ten restarts from the generator `default_rng(2021)` (a protocol constant, `generate.SPLIT_SEED`) and keeps the partition with the largest `zeta^2` at `x*`; rows within a client are in ascending original order. Restart 1 (zero-based, the second of the ten) is kept: `zeta^2` at `x*` is 0.0575932, against 0.0555548 for the worst restart and a total spread of 0.125264, so 46.0% of the upper bound (0.1253), against about 10% for the Dirichlet splits at `alpha` 0.1 and 0.01. K-means stops at a local optimum, so this is not the largest `zeta^2` any balanced split could have; the bound is only an upper limit. 23 of the 32 clients hold only negative examples, and the other 9 hold between 289 and 651 positive examples each.

**The partition is shipped.** The 32 x 1,000 row indices are tracked in `data/ijcnn1-32-kmeans-partition.json`, together with the SHA-256 of the partition as int64 values (row-major, clients in order), `lambda`, the number of restarts, the seed, the kept restart and the `zeta^2` of every restart. `generate.build('ijcnn1-32-kmeans')` loads this file, checks its SHA-256, the shape and that every row is held exactly once, and does not run k-means; `adafed/protocol.py` locks the digest. It is shipped because the construction is a long chain of floating-point decisions (the solution `x*`, the k-means++ draws, near-ties in the assignment) that a different BLAS or CPU can resolve differently in the last bits, which would give a different, equally valid partition and so different figures; a shipped partition makes the split, and with it every figure, independent of the machine. `python -m tools.kmeans_partition` (about two minutes) reruns the construction and reports whether it reproduces the tracked partition (exit status 0 if so); `--write` replaces the file. Regeneration on the machine that produced the file reproduces it exactly (same SHA-256, `b3ac05b09eecebaeb4fb4edf48c38a3bb8a47b3fe426f6192c5502b8434398e3`).

**Protocol.** Everything else is the protocol of the Dirichlet figures: tuning at `K=10` with seed 1901 from scratch (Figure 1: eta on the integer power-of-ten grid, 150 rounds, stochastic `b=1` tuned apart from exact; Figure 2: the joint eta and gamma grid for AdaFed and the method-specific grids for the baselines, `b=32`), grids extended while the pick is on an edge, then 10,000-round runs with full participation, exact seed 2027 once and stochastic seeds 2027--2046. The other figures follow the rate-study and benchmark protocol above, on the k-means datasets. Each k-means figure source (`<figure>-kmeans.tex`) is the default figure's source with its table names changed, the bands added and, for the fixed-eta figures, the eta labels set from the k-means runs (`tools/figure_sources.py`; checked by a test).

## The k-means splits of the synthetic datasets

Every synthetic dataset has a k-means split by the same construction as IJCNN1's (`generate.kmeans_split_info`: the
per-example data-loss gradients at the pooled L2 minimiser at `lambda = 0.01`, ten balanced k-means restarts from
`default_rng(2021)`, the restart with the largest `zeta^2` kept), into the default split's 32 clients of exactly 64 rows:
`synthetic-kappa1-kmeans`, `synthetic-kappa10-kmeans` and `synthetic-kappa100-kmeans`. The partitions are shipped in
`data/synthetic-kappa<k>-kmeans-partition.json` and their SHA-256 locked in `adafed/protocol.py`
(`kmeans.synthetic`); `python -m tools.kmeans_partition --dataset synthetic-kappa<k>` regenerates one. Their `F_star` (in
`data/fstar.json` and the lock) equal the default split's within one unit in the last place (at most `5.6e-17`).

With `--split kmeans` the whole pipeline runs on the k-means datasets: the rate study (exact `K` in `{1,10,100}` on all
four datasets, `b = 1` and the appendix batches, the eta, gamma and participation ablations at `K = 10`, the
10,000-round IJCNN1 `b = 1` panel) and the four benchmark groups on `synthetic-kappa1-kmeans` and `ijcnn1-32-kmeans`.
Everything is tuned from scratch on the k-means datasets with the default protocol (the synthetic eta at
`synthetic-kappa10-kmeans`, shared across the synthetic k-means datasets). The eta-sensitivity figures draw the default
figures' five values (synthetic `0.01..100`, IJCNN1 `0.1..1000`). The eta grid is centred on the largest client smoothness,
which the k-means split changes (IJCNN1: `10^0` instead of `10^1`, so its grid is `0.01..100`), so for the k-means split the
eta ablation runs every tuning candidate and also any of the figure's five values the grid lacks; tuning itself is
unchanged. The export stops unless every value the figure draws has finished runs for every seed.

`python -m tools.heterogeneity --synthetic` measures them like the IJCNN1 splits below (`tools/heterogeneity-synthetic.csv`).
As a fraction of the upper bound at the L2 `x*`, `zeta^2` is 1.5%, 1.6% and 1.8% for the default splits of `kappa` 1, 10
and 100, and 16.0%, 21.1% and 30.6% for their k-means splits (IJCNN1: 0.11% and 46.0%).

## Heterogeneity of the IJCNN1 splits

`python -m tools.heterogeneity` (from the repository root; about 30 s once the datasets exist, it builds the ones it needs) measures the default split, the Dirichlet splits `alpha` in `{0.01, 0.05, 0.1, 0.5, 1}` and the k-means split, adds a row with the upper bound of `zeta^2`, prints the table and writes `tools/heterogeneity-ijcnn1.csv`, which the table below reproduces (eight rows, one per split plus the bound). With `n_i` the client sizes, `n = 32,000`, `p_i` the fraction of positive labels of client `i` and `p_bar` the pooled fraction (0.09734):

- **Size**: the minimum, median and maximum of `n_i`.
- **Positives per client**: the minimum, median and maximum number of positive examples held by a client.
- **Label skew**: the weighted `sum_i (n_i/n) |p_i - p_bar|`, the unweighted mean of `|p_i - p_bar|`, and the number of single-class clients.
- **Gradient dissimilarity**: `zeta^2 = sum_i (n_i/n) ||grad l_i(x) - grad l(x)||^2`, with `l_i` client `i`'s mean logistic data loss (no penalty) and `l` the pooled data loss (the `n_i/n`-weighted mean of the `l_i`), evaluated at `x = 0` and at the minimiser `x*` of the smooth convex (L2) and of the nonsmooth convex (L1) problem (`lambda = 0.01`), each from the library's own solver (`problem.solve_l2sq`, `problem.solve_l1`) on the split's pooled rows. The pooled objective is the same on every split, so `x*` is the same up to the solver's round-off.
- **Upper bound**: the largest `zeta^2` any split can have at each of the three points, the total spread `(1/n) sum_j ||g_j - g_bar||^2` of the per-example data-loss gradients `g_j` around their mean `g_bar` (with the L2 and the L1 `x*` of the default split); `zeta^2` of a split is the between-client part of it, so it cannot exceed it.
- **Repair**: whether the Dirichlet construction's repair was used (no draw within the 1,000 redraws left every client non-empty) and the number of examples it moved.

| split | size min / median / max | positives per client min / median / max | label skew, weighted | label skew, unweighted | single-class clients | zeta^2 at x = 0 | zeta^2 at x* (L2) | zeta^2 at x* (L1) | repair (examples moved) |
|---|---|---|---|---|---|---|---|---|---|
| default (ijcnn1-32) | 1,000 / 1,000 / 1,000 | 78 / 98 / 111 | 0.0066 | 0.0066 | 0 | 4.137e-04 | 1.335e-04 | 1.395e-04 | no |
| Dirichlet 0.01 | 1 / 1 / 28,816 | 0 / 0 / 3,114 | 0.1757 | 0.1321 | 31 | 1.386e-02 | 1.347e-02 | 1.427e-02 | yes (26) |
| Dirichlet 0.05 | 1 / 17 / 15,869 | 0 / 0 / 2,457 | 0.1708 | 0.2830 | 27 | 1.229e-02 | 1.205e-02 | 1.278e-02 | yes (9) |
| Dirichlet 0.1 | 1 / 52.5 / 9,851 | 0 / 2 / 642 | 0.1578 | 0.5008 | 20 | 1.198e-02 | 1.193e-02 | 1.267e-02 | no |
| Dirichlet 0.5 | 5 / 619.5 / 7,810 | 0 / 43.5 / 1,184 | 0.1119 | 0.1790 | 1 | 6.564e-03 | 6.377e-03 | 6.767e-03 | no |
| Dirichlet 1 | 35 / 808 / 3,611 | 1 / 64.5 / 561 | 0.0656 | 0.0859 | 0 | 1.552e-03 | 1.425e-03 | 1.511e-03 | no |
| k-means (ijcnn1-32-kmeans) | 1,000 / 1,000 / 1,000 | 0 / 0 / 651 | 0.1399 | 0.1399 | 23 | 2.389e-01 | 5.759e-02 | 5.084e-02 | no |
| upper bound | | | | | | 3.144e-01 | 1.253e-01 | 1.288e-01 | |

As a fraction of the upper bound, `zeta^2` is 0.13%, 0.11% and 0.11% for the default split, 3.8%, 9.5% and 9.8% for Dirichlet 0.1, 4.4%, 10.8% and 11.1% for Dirichlet 0.01, and 76.0%, 46.0% and 39.5% for the k-means split (at `x = 0`, at `x*` of L2 and at `x*` of L1).

Alpha 0.01 is used in the figures: it is built with the repair (26 examples moved) and one client holds about 90% of the rows. Alpha 0.05 is measured but not used: its `zeta^2` is within about 1% of alpha 0.1's at the minimisers (2.6% at `x = 0`). The unweighted label skew is lower at alpha 0.01 and 0.05 than at 0.1 because most of the single-example clients are majority-class (negative) examples, which deviate from `p_bar` by only 0.097.
