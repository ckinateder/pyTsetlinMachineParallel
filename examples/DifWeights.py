"""
GrOTm: Gradient-Optimized Tsetlin Machine -- weight comparison experiment
==========================================================================

Compares four ways of getting clause weights onto a Tsetlin Machine's decision rule:
    - UTM:     unweighted TM (no weighting at all; every clause counts once)
    - WTM:     weighted TM using native online (TA-feedback) weight learning
    - WTM-NN:  WTM frozen partway through training, then a differentiable head
               (LogWeightHead) gradient-fits real-valued weights on its clause outputs
    - UTM-NN:  same as WTM-NN, but the frozen base TM is unweighted

For each dataset this reports, per method: per-epoch test accuracy, average
last-10-epoch test accuracy (the headline metric), training/inference time, paired
significance tests between each "-NN" method and its base TM, and a weight-resolution
ablation (does the -NN gain survive collapsing its real-valued weights onto the
positive-integer domain native training is restricted to?).

WHAT CHANGED IN THIS PASS (context if you're picking this back up):
    - Held-out validation split: weight-scaling calibration and NN model selection use
      x_val, never x_test (previously leaked through the test set).
    - Real per-run seeding (seed_everything) plus a per-run shuffled example order,
      since the C extension's own RNGs have no seed hook (see seed_everything's
      docstring) -- this is what gives the n_seeds runs genuine cross-seed variance.
    - Between-seed std in the aggregate CSV is computed correctly (one scalar per
      seed, then std across seeds), not pooled per-epoch values across seeds.
    - The "Cyclic" (TM<->NN alternating) method was removed entirely; see git history
      from before this pass if it needs to come back.
    - Added paired significance testing (aggregate_significance_tests: scipy paired
      t-test + Wilcoxon) and a weight-resolution ablation (weight_resolution_ablation /
      aggregate_weight_resolution_ablation).
    - Restructured the output directory layout (below) to separate per-seed detail
      from cross-seed rollups.
    - Plot colors are now a fixed, colorblind-validated 4-hue palette (METHOD_COLORS),
      not matplotlib's default cycle. Bar charts support error bars and significance
      brackets. Added a seed-averaged per-epoch accuracy chart (+/-1 std band), a
      weight-resolution-ablation chart (per-seed and aggregate), and a
      "{base_method}_baseline_test_accuracy" field on the ablation JSON/CSV so the
      continuous/int-rounded numbers can be read against the native TM's own accuracy
      without cross-referencing summary_results.csv. NOT backward compatible with
      weight_resolution_ablation.json files from before this field existed.
    - Added a wall-clock ("Accuracy vs. Training Time") variant of the per-epoch chart
      (per-seed and aggregate) -- an epoch is not a fixed unit of compute, so this is
      the fair "who gets further for equal compute" comparison the epoch-indexed
      chart can't give. Also added a "Shared checkpoint" marker on the epoch-indexed
      charts at split_one: WTM-NN/UTM-NN's TM phase literally reuses WTM's/UTM's own
      first-phase results, so the lines are identical data before that point.

OUTPUT LAYOUT:
    results/<dataset>_C<C>_T<T>_s<s>_e<total_epochs>/
        seed_0/ ... seed_<n-1>/   per-seed detail: per_epoch_results.csv,
                                  summary_results.csv, experiment_metadata.json,
                                  weight_resolution_ablation.json, seven plot_*.png
                                  (the two .pkl TM checkpoints are transient -- deleted
                                  once loaded, not left behind)
        aggregate/                cross-seed rollups: aggregate_summary_results.csv,
                                  significance_tests.csv,
                                  weight_resolution_ablation_aggregate.csv,
                                  six plot_aggregate_*.png

HOW TO RUN:
    Must run inside the project's Docker container -- the C extension needs a Linux +
    OpenMP build environment (see CLAUDE.md). From the repo root:
        docker build -t pytsetlin .
        docker run --rm -it -v $(pwd):$(pwd) pytsetlin bash
    The Dockerfile installs requirements.txt and builds the package at image-build
    time, so a container from that image works out of the box. If you bind-mount your
    own working copy over it for active development, re-run `pip install -e .` once
    inside the container so the extension rebuilds against your live source.

    Then, from the repo root inside the container:
        python examples/DifWeights.py
    Runs the full sweep configured in __main__ (4 datasets x 5 seeds x 500 epochs by
    default -- expect hours; a single 5-seed dataset config is on the order of tens of
    minutes). A (dataset, config, seed) run already on disk (per_epoch_results.csv +
    summary_results.csv present) is skipped and just replotted on a re-run.
"""

import json
import logging
import random
from datetime import datetime, timezone
from time import perf_counter
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from torchvision.datasets import MNIST, EMNIST, KMNIST, FashionMNIST
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import os
import warnings
from copy import deepcopy
import itertools
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Console logging: timestamped, one line per phase/status event -- distinct from the
# plain print() calls used for tabular data dumps (DataFrame.to_string()), which stay
# untimestamped since a timestamp on a table header doesn't help read the table.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("DifWeights")

# Per-epoch training metrics returned by train_tm / train_log_weight_head.
# Each list element is one epoch, in order. Keys are floats (seconds for times).
EpochResult = dict[str, float]
# Keys: "test_accuracy" (percent 0-100), "test_time" (s), "train_time" (s).

THRESHOLD = 75

# Summary table / aggregate order (also default row order in summary CSV).
METHODS_ORDERED = ("UTM", "WTM", "WTM-NN", "UTM-NN")

# Per-epoch curve legend and color assignment (intentionally not METHODS_ORDERED).
METHODS_PLOT_ORDER = ("WTM-NN", "UTM-NN", "UTM", "WTM")

# Pairs for paired significance testing: (treatment, base), matched by seed index.
# Seed i's treatment run and seed i's base run share the same shuffled train/val split
# and the same epoch-split_one checkpoint (see run_experiment), so pairing by seed
# index -- not independently sorted/grouped values -- is the meaningful pairing.
SIGNIFICANCE_PAIRS: tuple[tuple[str, str], ...] = (("WTM-NN", "WTM"), ("UTM-NN", "UTM"))
# treatment method -> base method (e.g. "WTM-NN" -> "WTM"), reused wherever a baseline
# needs to be labeled by the base model's own name rather than the -NN treatment's.
BASE_METHOD_OF: dict[str, str] = dict(SIGNIFICANCE_PAIRS)

# Base seed; run i uses BASE_SEED + i (see __main__).
BASE_SEED = 1000


def seed_everything(seed: int) -> None:
    """Seed Python / NumPy / Torch RNGs for one experiment run.

    NOTE: the C extension's RNGs (fast_rand PCG in fast_rand.h, libc rand() in
    mc_tm_update) have no seed hook, and mc_tm_fit runs OpenMP-parallel with lock
    races on clause updates. TM training is therefore NOT bitwise-reproducible from
    Python. Cross-seed variance in the TM comes from the per-run shuffled example
    order (see run_experiment), not from these RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_summary_df(per_epoch_df: pd.DataFrame) -> pd.DataFrame:
    """One row per METHODS_ORDERED; must match run_experiment summary CSV columns."""
    summary_rows = []
    for method in METHODS_ORDERED:
        mdf = per_epoch_df[per_epoch_df["method"] == method]
        summary_rows.append({
            "method": method,
            "avg_last10_tm_accuracy": round(float(mdf["test_accuracy"].tail(10).mean()), 2),
            "avg_tm_epoch_time_s": round(mdf[mdf["model_type"] == "tm"]["train_time"].mean(), 3),
            "avg_last10_tm_test_time_s": round(mdf["test_time"].tail(10).mean(), 4),
            "total_train_time_s": round(mdf["train_time"].sum(), 2),
        })
    return pd.DataFrame(summary_rows)


def binarize_dataset(train: Dataset, test: Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    x_train = np.where(train.data.numpy() > THRESHOLD, 1, 0).reshape(-1, 28 * 28).astype(np.uint32)
    y_train = train.targets.numpy().astype(np.uint32)
    x_test = np.where(test.data.numpy() > THRESHOLD, 1, 0).reshape(-1, 28 * 28).astype(np.uint32)
    y_test = test.targets.numpy().astype(np.uint32)

    mn = np.min(y_train)
    assert mn == np.min(y_test), "Min label mismatch"
    y_train = y_train - mn
    y_test  = y_test  - mn

    return x_train, y_train, x_test, y_test

def train_tm(
    model: MultiClassTsetlinMachine,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
) -> Tuple[list[EpochResult], MultiClassTsetlinMachine]:

    results: list[EpochResult] = []
    pbar = tqdm(range(epochs), desc=f"TM", dynamic_ncols=True, leave=False)
    for _ in pbar:
        train_start = perf_counter()
        model.fit(x_train, y_train, epochs=1, incremental=True)
        train_elapsed = perf_counter() - train_start
        test_start = perf_counter()
        acc = 100.0 * (model.predict(x_test) == y_test).mean()
        test_elapsed = perf_counter() - test_start
        results.append(
            {
                "test_accuracy": acc,
                "test_time": test_elapsed,
                "train_time": train_elapsed,
            }
        )
        pbar.set_postfix(acc=f"{results[-1]['test_accuracy']:.2f}%")
    pbar.close()
    return results

class LogWeightHead(nn.Module):
    """
    Differentiable positive clause-weight head for a frozen multiclass TM.

    Input:
        Z shape = [batch_size, n_classes * n_clauses]

    Internally:
        Z is reshaped to [batch_size, n_classes, n_clauses]

    Output:
        logits shape = [batch_size, n_classes]

    Logit for class k:
        logit_k = bias_k + sum_j polarity_j * exp(theta_kj) * clause_output_kj
    """
    def __init__(self, n_classes, n_clauses, T: float = None, init_weights=None):
        super().__init__()

        self.n_classes = n_classes
        self.n_clauses = n_clauses
        self.T = T
        
        if init_weights is not None and T is not None:
            logger.warning("LogWeightHead: both init_weights and T are provided - this clamps gradients")
        
        if init_weights is None:
            theta_init = torch.zeros(n_classes, n_clauses)
        else:
            init_weights = torch.as_tensor(init_weights, dtype=torch.float32)
            init_weights = torch.clamp(init_weights, min=1e-6)
            theta_init = torch.log(init_weights)

        self.theta = nn.Parameter(theta_init)

        self.register_buffer("bias", torch.zeros(n_classes))

        polarity = torch.ones(n_clauses)
        polarity[1::2] = -1.0
        self.register_buffer("polarity", polarity)


    @property
    def weights(self) -> torch.Tensor:
        return torch.exp(self.theta)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: [B, K*C]
        # reshapes from (n_examples, n_classes*n_clauses) to (n_examples, n_classes, n_clauses)
        Z = Z.float().view(-1, self.n_classes, self.n_clauses)

        # [K, C]
        # apply polarity, since the states from the TM are unsigned        
        signed_weights = self.weights * self.polarity.view(1, -1)

        # [B, K]
        # broadcast the same class-clause weights to all examples
        logits = (Z * signed_weights.unsqueeze(0)).sum(dim=2) + self.bias

        if self.T is not None:
            logits = logits.clamp(-self.T, self.T)

        return logits

def scale_weights_for_tm(model: LogWeightHead, Z: np.ndarray, T: float, safety: float = 0.95) -> np.ndarray:
    """Scale NN weights so max raw vote <= T*safety, preventing TM clamp from activating."""
    device = next(model.parameters()).device
    with torch.no_grad():
        logits = model(torch.as_tensor(Z, dtype=torch.uint8).to(device))
        max_logit = logits.abs().max().item()
    alpha = (T * safety) / max_logit
    return model.weights.detach().cpu().numpy() * alpha


def evaluate_weights_direct(tm: MultiClassTsetlinMachine, weights: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Set `weights` directly on a copy of `tm` (no scale_weights_for_tm rescaling) and
    return test accuracy. Mirrors how native weighted_clauses training uses its own
    learned weights at inference: raw magnitudes, whatever clamp behavior falls out
    of them -- not the clamp-avoidance rescaling the main pipeline applies to NN weights.
    """
    eval_tm = deepcopy(tm)
    eval_tm.set_clause_weights(weights)
    return 100.0 * (eval_tm.predict(x) == y).mean()


def integer_round_weights(weights: np.ndarray) -> np.ndarray:
    """Round to the nearest integer >= 1 -- the domain (positive integers, floor 1)
    native weighted_clauses training is restricted to by its +/-1.0 update rule
    (see ConvolutionalTsetlinMachine.c). Used by weight_resolution_ablation below."""
    return np.maximum(np.round(weights), 1.0).astype(np.float32)


def weight_resolution_ablation(
    method: str,
    nn_model: LogWeightHead,
    frozen_tm: MultiClassTsetlinMachine,
    x_test: np.ndarray,
    y_test: np.ndarray,
    base_method: str,
    baseline_test_accuracy: float,
) -> dict:
    """
    Isolates weight DOMAIN (continuous reals vs. the positive-integer, floor-1
    lattice native weighted_clauses training is restricted to) from the choice of
    OPTIMIZER (gradient descent vs. online TA feedback) -- see the "GrOTm" writeup
    discussion on the continuous-vs-integer confound. Both numbers below go through
    the identical evaluate_weights_direct path (no scale_weights_for_tm rescale);
    rounding is the only difference between them, so any accuracy gap is
    attributable to weight resolution alone. If int_rounded accuracy holds up close
    to continuous, the method's gain is not merely from higher numeric precision.

    base_method / baseline_test_accuracy: the native TM's own name ("WTM"/"UTM", not
    the "-NN" treatment name) and avg-last-10 test accuracy (same rule as
    compute_summary_df) -- gives the continuous/int-rounded numbers something to be
    read against without cross-referencing summary_results.csv. Keyed by base_method
    (e.g. "WTM_baseline_test_accuracy") since the baseline IS the base model's own
    accuracy, not a property of the "-NN" method.
    """
    raw_weights = nn_model.weights.detach().cpu().numpy()
    continuous_acc = evaluate_weights_direct(frozen_tm, raw_weights, x_test, y_test)
    int_rounded_acc = evaluate_weights_direct(frozen_tm, integer_round_weights(raw_weights), x_test, y_test)
    return {
        f"{base_method}_baseline_test_accuracy": baseline_test_accuracy,
        f"{method}_direct_continuous_test_accuracy": continuous_acc,
        f"{method}_direct_int_rounded_test_accuracy": int_rounded_acc,
        f"{method}_weight_resolution_gap": continuous_acc - int_rounded_acc,
    }


def train_log_weight_head(
    model: LogWeightHead,
    tm: MultiClassTsetlinMachine,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int = 512,
    learning_rate: float = 1e-2,
    l2_weight: float = 1e-5,
    patience: int = 1000,
) -> list[EpochResult]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    z_train = tm.transform(x_train, inverted=False).astype("uint8")
    z_val   = tm.transform(x_val,   inverted=False).astype("uint8")
    z_test  = tm.transform(x_test,  inverted=False).astype("uint8")

    z_train_t = torch.as_tensor(z_train, dtype=torch.uint8)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)

    loader = DataLoader(TensorDataset(z_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    # weight_decay=0: we apply manual L2 toward theta_init below, not toward zero.
    # This keeps the NN from inflating dominant clause weights at the expense of informative small ones.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    theta_init = model.theta.data.clone().to(device)

    # Eval TM: a copy used only for per-epoch TM accuracy measurement (never trained).
    eval_tm = deepcopy(tm)

    results: list[EpochResult] = []
    best_acc = 0.0
    best_theta = model.theta.data.clone()
    epochs_no_improve = 0
    pbar = tqdm(range(epochs), desc=f"NN", dynamic_ncols=True, leave=False)
    for epoch in pbar:
        train_start = perf_counter()
        for zb, yb in loader:
            zb, yb = zb.to(device), yb.to(device)
            loss = F.cross_entropy(model(zb), yb) + l2_weight * (model.theta - theta_init).pow(2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_elapsed = perf_counter() - train_start

        # Scale NN weights using held-out val activations (never the test set), apply
        # to eval_tm, then measure accuracy. best_theta / early stopping track VAL
        # accuracy; the per-epoch test accuracy is recorded only for the reported
        # learning curve, never for model selection. Only the test predict is timed
        # (test_time = deployed inference cost); the val predict is a training concern.
        scaled = scale_weights_for_tm(model, z_val, tm.T)
        eval_tm.set_clause_weights(scaled)
        val_acc = 100.0 * (eval_tm.predict(x_val) == y_val).mean()
        test_start = perf_counter()
        acc = 100.0 * (eval_tm.predict(x_test) == y_test).mean()
        test_elapsed = perf_counter() - test_start

        results.append({"test_accuracy": acc, "test_time": test_elapsed, "train_time": train_elapsed})
        if val_acc > best_acc:
            best_acc = val_acc
            best_theta = model.theta.data.clone()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        pbar.set_postfix(val_acc=f"{val_acc:.2f}%", tm_acc=f"{acc:.2f}%", loss=f"{loss.item():.2f}")

        if epochs_no_improve >= patience:
            break

    model.theta.data.copy_(best_theta)
    return results

def _std(vals: list[float]) -> float:
    """Sample std, ddof=1. Returns 0.0 for <2 values (not NaN) -- "one observation, no
    spread" is a meaningful state for the between-seed rollups that use this."""
    if len(vals) < 2:
        return 0.0
    return float(np.std(vals, ddof=1))


def aggregate_experiment_results(
    dataset_name: str,
    per_epoch_dfs: list[pd.DataFrame],
    summary_dfs: list[pd.DataFrame],
    save_path: str,
    *,
    total_epochs: int | None = None,
    T: int | None = None,
    s: float | None = None,
    split_point: float | None = None,
    val_fraction: float | None = None,
) -> pd.DataFrame:
    """
    Per method: mean and sample std (ddof=1) over the n_seeds runs.
    Every metric is reduced to one scalar per run first, so the reported std is the
    between-seed std (n_seeds values), not a pooled within-run/between-run mixture:
    - avg_last10_tm_accuracy: mean of the last 10 test_accuracy rows per method/run
      (same rule as compute_summary_df), then mean/std across runs.
    - avg_tm_epoch_time_s, avg_last10_tm_test_time_s, total_train_time_s: recomputed / read like
      run_experiment summary (one scalar per run), then mean/std across runs.

    Optional kwargs total_epochs, T, s, split_point, val_fraction are copied into every
    output row (same experiment config for all seeds); omit or pass None to leave those
    CSV cells blank.
    """
    n = len(per_epoch_dfs)
    if n == 0:
        raise ValueError("aggregate_experiment_results: no runs")
    if len(summary_dfs) != n:
        raise ValueError("aggregate_experiment_results: per_epoch_dfs and summary_dfs length mismatch")

    out_rows = []
    for method in METHODS_ORDERED:
        acc_vals: list[float] = []
        tm_train_vals: list[float] = []
        inf_vals: list[float] = []
        tot_vals: list[float] = []
        for pe_df, su_df in zip(per_epoch_dfs, summary_dfs):
            mdf = pe_df[pe_df["method"] == method]
            # One scalar per run: the run's own avg-last-10 test accuracy.
            acc_vals.append(float(mdf["test_accuracy"].tail(10).mean()))
            tm_train_vals.append(float(mdf[mdf["model_type"] == "tm"]["train_time"].mean()))
            srow = su_df[su_df["method"] == method].iloc[0]
            inf_vals.append(float(srow["avg_last10_tm_test_time_s"]))
            tot_vals.append(float(srow["total_train_time_s"]))

        out_rows.append({
            "dataset": dataset_name,
            "total_epochs": total_epochs,
            "T": T,
            "s": s,
            "split_point": split_point,
            "val_fraction": val_fraction,
            "method": method,
            "n_seeds": n,
            "avg_last10_tm_accuracy_mean": round(float(np.mean(acc_vals)), 4),
            "avg_last10_tm_accuracy_std": round(_std(acc_vals), 4),
            "avg_tm_epoch_time_s_mean": round(float(np.mean(tm_train_vals)), 4),
            "avg_tm_epoch_time_s_std": round(_std(tm_train_vals), 4),
            "avg_last10_tm_test_time_s_mean": round(float(np.mean(inf_vals)), 4),
            "avg_last10_tm_test_time_s_std": round(_std(inf_vals), 4),
            "total_train_time_s_mean": round(float(np.mean(tot_vals)), 2),
            "total_train_time_s_std": round(_std(tot_vals), 2),
        })

    agg_df = pd.DataFrame(out_rows)
    os.makedirs(save_path, exist_ok=True)  # save_path is a per-config "aggregate/" dir, may not exist yet
    out_path = os.path.join(save_path, AGGREGATE_SUMMARY_RESULTS_FILENAME)
    agg_df.to_csv(out_path, index=False)
    logger.info(f"Aggregate summary (mean +/- between-seed std over {n} seeds) saved to {out_path}")
    print(agg_df.to_string(index=False))
    return agg_df


def aggregate_significance_tests(
    dataset_name: str,
    per_epoch_dfs: list[pd.DataFrame],
    save_path: str,
    *,
    pairs: tuple[tuple[str, str], ...] = SIGNIFICANCE_PAIRS,
    total_epochs: int | None = None,
    T: int | None = None,
    s: float | None = None,
) -> pd.DataFrame:
    """
    Paired significance testing between each (treatment, base) method pair in `pairs`,
    matched by seed index: seed i's treatment run and seed i's base run share the same
    shuffled train/val split and the same epoch-split_one checkpoint (see run_experiment),
    so pairing by seed index -- not independently sorted/grouped values -- is the
    statistically meaningful pairing.

    Per pair: reduces each seed's per_epoch_df to one avg-last-10 test_accuracy scalar per
    method (same reduction as compute_summary_df / aggregate_experiment_results), then runs:
      - scipy.stats.ttest_rel: paired t-test, the primary/more informative statistic here.
      - scipy.stats.wilcoxon: paired non-parametric test, secondary robustness check only.
        NOTE: with n_seeds=5 the minimum achievable two-sided wilcoxon p-value is 0.0625
        (2**-4) -- it can never report p < 0.05 at this sample size, so treat it as
        supporting evidence, not a pass/fail gate.
    Also reports the raw paired differences (mean_diff, std_diff; ddof=1 via the
    module-level _std) since at n=5 that is arguably more informative than either p-value.

    Degenerate-input handling (must not crash a multi-hour sweep):
      - n_pairs < 2: both tests are skipped; p-values are NaN, wilcoxon_note explains why.
      - ttest_rel on zero-variance differences returns NaN statistic/p-value without
        raising (scipy's normal behavior, possibly with a RuntimeWarning) -- warnings are
        suppressed and the NaN is written as-is.
      - wilcoxon raises ValueError when all paired differences are exactly zero (the
        default zero-handling method requires at least one non-zero diff) -- caught,
        wilcoxon_statistic/pvalue left NaN, wilcoxon_note set to the exception message.

    Writes one row per pair to {save_path}/significance_tests.csv with columns:
        dataset, total_epochs, T, s, treatment_method, base_method, n_pairs,
        treatment_mean, base_mean, mean_diff, std_diff,
        ttest_statistic, ttest_pvalue, wilcoxon_statistic, wilcoxon_pvalue, wilcoxon_note
    """
    os.makedirs(save_path, exist_ok=True)
    rows = []
    for treatment, base in pairs:
        treat_vals, base_vals = [], []
        for pe_df in per_epoch_dfs:
            treat_vals.append(float(pe_df[pe_df["method"] == treatment]["test_accuracy"].tail(10).mean()))
            base_vals.append(float(pe_df[pe_df["method"] == base]["test_accuracy"].tail(10).mean()))
        treat_arr, base_arr = np.array(treat_vals), np.array(base_vals)
        diffs = treat_arr - base_arr
        n_pairs = len(diffs)

        ttest_stat, ttest_p = float("nan"), float("nan")
        wilcoxon_stat, wilcoxon_p, wilcoxon_note = float("nan"), float("nan"), ""
        if n_pairs >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tres = stats.ttest_rel(treat_arr, base_arr)
            ttest_stat, ttest_p = float(tres.statistic), float(tres.pvalue)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wres = stats.wilcoxon(treat_arr, base_arr)
                wilcoxon_stat, wilcoxon_p = float(wres.statistic), float(wres.pvalue)
            except ValueError as e:
                wilcoxon_note = str(e)
        else:
            wilcoxon_note = "n_pairs < 2, skipped"

        rows.append({
            "dataset": dataset_name, "total_epochs": total_epochs, "T": T, "s": s,
            "treatment_method": treatment, "base_method": base, "n_pairs": n_pairs,
            "treatment_mean": round(float(np.mean(treat_arr)), 4) if n_pairs else float("nan"),
            "base_mean": round(float(np.mean(base_arr)), 4) if n_pairs else float("nan"),
            "mean_diff": round(float(np.mean(diffs)), 4) if n_pairs else float("nan"),
            "std_diff": round(_std(list(diffs)), 4),
            "ttest_statistic": ttest_stat, "ttest_pvalue": ttest_p,
            "wilcoxon_statistic": wilcoxon_stat, "wilcoxon_pvalue": wilcoxon_p,
            "wilcoxon_note": wilcoxon_note,
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(save_path, SIGNIFICANCE_TESTS_FILENAME)
    df.to_csv(out_path, index=False)
    logger.info(f"Significance tests saved to {out_path}")
    print(df.to_string(index=False))
    return df


def aggregate_weight_resolution_ablation(
    dataset_name: str,
    ablations: list[dict | None],
    save_path: str,
    *,
    total_epochs: int | None = None,
    T: int | None = None,
    s: float | None = None,
) -> pd.DataFrame:
    """
    Aggregates per-seed weight_resolution_ablation dicts (see weight_resolution_ablation()
    and run_experiment's 3rd return value) across seeds, mean/std (ddof=1, via the
    module-level _std) per method in ("WTM-NN", "UTM-NN"). Seeds whose ablation entry is
    None are skipped (e.g. a cache-hit run whose weight_resolution_ablation.json is
    missing on disk) -- n_available reports how many of n_seeds_total actually
    contributed, so a partially-populated aggregate stays visible rather than silently
    averaging fewer seeds than the run count implies.

    NOTE: unlike _std's own 0.0-for-<2-values convention (kept as-is here too when at
    least one seed is available), an EMPTY list here (n_available == 0) reports NaN
    rather than 0.0 for every mean/std column, since "zero seeds had ablation data" is a
    materially different, worse state than "one seed had ablation data with std 0" and
    collapsing them to the same 0.0 would be misleading.

    Writes one row per method to {save_path}/weight_resolution_ablation_aggregate.csv with
    columns:
        dataset, total_epochs, T, s, method, n_seeds_total, n_available,
        baseline_test_accuracy_mean, baseline_test_accuracy_std,
        direct_continuous_test_accuracy_mean, direct_continuous_test_accuracy_std,
        direct_int_rounded_test_accuracy_mean, direct_int_rounded_test_accuracy_std,
        weight_resolution_gap_mean, weight_resolution_gap_std

    NOT backward compatible with weight_resolution_ablation.json files predating the
    "{base_method}_baseline_test_accuracy" field (raises KeyError) -- delete stale
    per-seed dirs and rerun rather than mixing old and new ablation JSONs in one
    aggregate.
    """
    os.makedirs(save_path, exist_ok=True)
    n_total = len(ablations)
    available = [a for a in ablations if a is not None]

    def _mean_or_nan(vals: list[float]) -> float:
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    def _std_or_nan(vals: list[float]) -> float:
        return round(_std(vals), 4) if vals else float("nan")

    rows = []
    for method in ("WTM-NN", "UTM-NN"):
        baseline_vals = [a[f"{BASE_METHOD_OF[method]}_baseline_test_accuracy"] for a in available]
        cont_vals = [a[f"{method}_direct_continuous_test_accuracy"] for a in available]
        rounded_vals = [a[f"{method}_direct_int_rounded_test_accuracy"] for a in available]
        gap_vals = [a[f"{method}_weight_resolution_gap"] for a in available]
        rows.append({
            "dataset": dataset_name, "total_epochs": total_epochs, "T": T, "s": s,
            "method": method, "n_seeds_total": n_total, "n_available": len(cont_vals),
            "baseline_test_accuracy_mean": _mean_or_nan(baseline_vals),
            "baseline_test_accuracy_std": _std_or_nan(baseline_vals),
            "direct_continuous_test_accuracy_mean": _mean_or_nan(cont_vals),
            "direct_continuous_test_accuracy_std": _std_or_nan(cont_vals),
            "direct_int_rounded_test_accuracy_mean": _mean_or_nan(rounded_vals),
            "direct_int_rounded_test_accuracy_std": _std_or_nan(rounded_vals),
            "weight_resolution_gap_mean": _mean_or_nan(gap_vals),
            "weight_resolution_gap_std": _std_or_nan(gap_vals),
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(save_path, WEIGHT_RESOLUTION_ABLATION_AGGREGATE_FILENAME)
    df.to_csv(out_path, index=False)
    n_avail = len(available)
    logger.info(f"Weight-resolution ablation aggregate saved to {out_path} ({n_avail}/{n_total} seeds had ablation data)")
    print(df.to_string(index=False))
    return df


EXPERIMENT_METADATA_FILENAME = "experiment_metadata.json"
EXPERIMENT_METADATA_SCHEMA_VERSION = 3

# Seven PNGs per seed (no results.png): four singles + one 2x2 combined + one
# ablation + one wall-clock accuracy (epoch is not a fixed unit of compute).
PLOT_PER_EPOCH_ACCURACY_PNG = "plot_per_epoch_accuracy.png"
PLOT_PER_EPOCH_ACCURACY_BY_TIME_PNG = "plot_per_epoch_accuracy_by_time.png"
PLOT_AVG_LAST10_ACCURACY_PNG = "plot_avg_last10_accuracy.png"
PLOT_AVG_LAST10_TM_TEST_TIME_PNG = "plot_avg_last10_tm_test_time.png"
PLOT_TOTAL_TRAIN_TIME_PNG = "plot_total_train_time.png"
PLOT_COMBINED_PNG = "plot_combined.png"
PLOT_WEIGHT_RESOLUTION_ABLATION_PNG = "plot_weight_resolution_ablation.png"

# Six PNGs per dataset, written once after the seed loop by write_aggregate_plots.
PLOT_AGGREGATE_PER_EPOCH_ACCURACY_PNG = "plot_aggregate_per_epoch_accuracy.png"
PLOT_AGGREGATE_PER_EPOCH_ACCURACY_BY_TIME_PNG = "plot_aggregate_per_epoch_accuracy_by_time.png"
PLOT_AGGREGATE_ACCURACY_PNG = "plot_aggregate_accuracy.png"
PLOT_AGGREGATE_TEST_TIME_PNG = "plot_aggregate_test_time.png"
PLOT_AGGREGATE_TRAIN_TIME_PNG = "plot_aggregate_train_time.png"
PLOT_AGGREGATE_WEIGHT_RESOLUTION_PNG = "plot_aggregate_weight_resolution.png"

# Output-tree filename/dirname constants (see run_experiment / __main__ for the layout):
#   results/<dataset>_C<C>_T<T>_s<s>_e<total_epochs>/seed_<i>/...       per-seed detail
#   results/<dataset>_C<C>_T<T>_s<s>_e<total_epochs>/aggregate/...      cross-seed rollups
WEIGHT_RESOLUTION_ABLATION_FILENAME = "weight_resolution_ablation.json"
AGGREGATE_DIRNAME = "aggregate"
SEED_DIR_PREFIX = "seed_"
AGGREGATE_SUMMARY_RESULTS_FILENAME = "aggregate_summary_results.csv"
SIGNIFICANCE_TESTS_FILENAME = "significance_tests.csv"
WEIGHT_RESOLUTION_ABLATION_AGGREGATE_FILENAME = "weight_resolution_ablation_aggregate.csv"

_SERIF_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
}

# Fixed, validated categorical palette -- NOT derived from matplotlib's rcParams cycle.
# Checked with the dataviz-skill color validator (node scripts/validate_palette.js
# "#2a78d6,#eb6834,#1baf7a,#eda100" --mode light): PASS on lightness band, chroma
# floor, CVD separation (worst adjacent dE 9.1), normal-vision floor (worst adjacent
# dE 22.9). WARN on raw contrast-vs-surface for aqua/yellow is satisfied by the
# "relief rule" -- every chart here already carries direct bar labels and/or a legend.
METHOD_COLORS: dict[str, str] = {
    "WTM-NN": "#2a78d6",  # blue
    "UTM-NN": "#eb6834",  # orange
    "UTM":    "#1baf7a",  # aqua
    "WTM":    "#eda100",  # yellow
}

# Weight-resolution-ablation chart colors: "baseline" is a neutral reference point
# (gray), "continuous" vs "int-rounded" is ordinal (same NN result, full precision vs
# quantized) so it's one hue in two lightness steps, not two unrelated categorical hues.
ABLATION_BASELINE_COLOR = "#999999"
ABLATION_CONTINUOUS_COLOR = "#4a3aa7"  # hue slot 7 (violet); int-rounded reuses this at:
ABLATION_INT_ROUNDED_ALPHA = 0.5       # ...half alpha, same hex, edgecolor kept solid.


def _plot_color_map() -> dict[str, str]:
    return dict(METHOD_COLORS)


def _hparam_legend_matching_bbox() -> dict:
    """Text bbox props aligned with the default legend frame (same rcParams as ax.legend)."""
    rp = plt.rcParams
    fc = rp["legend.facecolor"]
    if fc == "inherit":
        fc = rp["axes.facecolor"]
    pad = float(rp["legend.borderpad"])
    boxstyle = f"round,pad={pad}" if rp["legend.fancybox"] else f"square,pad={pad}"
    return {
        "boxstyle": boxstyle,
        "facecolor": fc,
        "edgecolor": rp["legend.edgecolor"],
        "linewidth": float(rp["axes.linewidth"]),
        "alpha": float(rp["legend.framealpha"]),
    }


def _annotate_hyperparams_bottom_right(ax, meta: dict) -> None:
    """C, T, s in axes lower-right (left-justified block; anchor is lower-left of text)."""
    text = f"C = {meta['C']}\nT = {meta['T']}\ns = {meta['s']}"
    ax.text(
        0.9, 0.02, text, transform=ax.transAxes, fontsize=9,
        horizontalalignment="left", verticalalignment="bottom", multialignment="left", bbox=_hparam_legend_matching_bbox(),
    )


def _annotate_hyperparams_epoch_left_of_legend(ax, meta: dict) -> None:
    """C, T, s left-justified, lower area to the left of the lower-right legend."""
    text = f"C = {meta['C']}\nT = {meta['T']}\ns = {meta['s']}"
    ax.text(
        0.72, 0.02, text, transform=ax.transAxes, fontsize=9,
        horizontalalignment="left", verticalalignment="bottom", multialignment="left", bbox=_hparam_legend_matching_bbox(),
    )


def _draw_per_epoch_accuracy_ax(
    ax, per_epoch_df: pd.DataFrame, color_map: dict[str, str], *,
    std_col: str | None = None, x_col: str = "epoch", x_label: str = "Epoch",
) -> None:
    """
    std_col, when given, shades a +/-1 sample-std band (not SEM, not a CI -- matches
    this module's mean +/- std convention everywhere else) around each segment's line,
    using a column already present on per_epoch_df (see _aggregate_per_epoch). No band
    is drawn on the trailing dotted flat-continuation below, since that segment is an
    extrapolated placeholder, not real per-epoch data.

    x_col/x_label let the same line-drawing logic serve either the epoch-indexed chart
    (x_col="epoch", the default) or the wall-clock chart (x_col="cumulative_train_time_s",
    see _add_cumulative_time) -- an epoch is not a fixed unit of compute (a TM epoch
    costs far more than an NN epoch), so the wall-clock variant is the one to trust for
    an apples-to-apples "who gets further for the same compute" comparison.
    """
    max_x = per_epoch_df[x_col].max()
    for method in METHODS_PLOT_ORDER:
        color = color_map[method]
        train_df = per_epoch_df[per_epoch_df["method"] == method].sort_values(x_col).reset_index(drop=True)
        if train_df.empty:
            continue
        train_df = train_df.copy()
        train_df["segment"] = (train_df["model_type"] != train_df["model_type"].shift()).cumsum()
        first = True
        for seg_id, seg in train_df.groupby("segment"):
            ls = "--" if seg["model_type"].iloc[0] == "nn" else "-"
            if seg_id > train_df["segment"].min():
                prev = train_df[train_df["segment"] == seg_id - 1].iloc[-1]
                xs = [prev[x_col]] + list(seg[x_col])
                accs = [prev["test_accuracy"]] + list(seg["test_accuracy"])
                stds = [prev[std_col]] + list(seg[std_col]) if std_col is not None else None
            else:
                xs = list(seg[x_col])
                accs = list(seg["test_accuracy"])
                stds = list(seg[std_col]) if std_col is not None else None
            ax.plot(xs, accs, color=color, linestyle=ls, linewidth=1.5, label=method if first else "_nolegend_")
            if stds is not None:
                lo = [a - s for a, s in zip(accs, stds)]
                hi = [a + s for a, s in zip(accs, stds)]
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
            first = False
        last = train_df.iloc[-1]
        if last[x_col] < max_x:
            ax.plot(
                [last[x_col], max_x], [last["test_accuracy"]] * 2,
                color=color, linestyle=":", linewidth=1.0, alpha=0.4,
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Per-Epoch Accuracy")
    ax.grid(True, alpha=0.3, zorder=0)
    method_handles = [Line2D([0], [0], color=color_map[m], linewidth=1.5, label=m) for m in METHODS_PLOT_ORDER]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="TM training"),
        Line2D([0], [0], color="black", linestyle="--", label="NN training"),
    ]
    ax.legend(handles=method_handles + style_handles, fontsize=8, loc="lower right")


def _add_cumulative_time(per_epoch_df: pd.DataFrame, time_col: str = "train_time") -> pd.DataFrame:
    """
    Adds cumulative_train_time_s: per-method cumulative sum of time_col, sorted by
    epoch. An epoch is not a fixed unit of compute -- a TM epoch (OpenMP clause/TA
    updates) costs far more wall-clock than an NN epoch (a small linear head over
    precomputed clause outputs) -- so plotting accuracy against actual elapsed
    training time, not epoch index, is the fair "who gets further for equal compute"
    comparison. Works on both a per-seed per_epoch_df (has train_time directly) and
    an _aggregate_per_epoch result (has a mean train_time column of the same name).
    """
    df = per_epoch_df.sort_values(["method", "epoch"]).copy()
    df["cumulative_train_time_s"] = df.groupby("method")[time_col].cumsum()
    return df


def _annotate_checkpoint(ax, x_value: float, label: str = "Shared checkpoint") -> None:
    """
    Vertical marker at the point WTM-NN/UTM-NN branch off from the still-training
    WTM/UTM. Both lines are literally the same data before this point -- WTM-NN's TM
    phase reuses WTM's own first-phase per-epoch results (same trained object) -- so
    this makes explicit, rather than leaving the reader to notice, that the two
    methods start from an identical shared state.
    """
    ylim = ax.get_ylim()
    ax.axvline(x_value, color="black", linestyle=":", linewidth=1.0, alpha=0.5)
    ax.text(x_value, ylim[0] + (ylim[1] - ylim[0]) * 0.02, f" {label}", ha="left", va="bottom",
             fontsize=8, alpha=0.7, rotation=90)


def _significance_stars(p: float) -> str:
    """*** p<0.001, ** p<0.01, * p<0.05, else "ns" (also "ns" for NaN, which
    aggregate_significance_tests legitimately produces for n_pairs < 2)."""
    if not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _draw_significance_bracket(ax, x1: float, x2: float, y: float, label: str) -> None:
    """Publication-style bracket: horizontal line spanning x1..x2 at height y, short
    vertical tick-downs at each end, label centered above."""
    tick = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    ax.plot([x1, x1, x2, x2], [y - tick, y, y, y - tick], color="black", linewidth=1.0)
    ax.text((x1 + x2) / 2, y, label, ha="center", va="bottom", fontsize=9)


def _draw_bar_metric_ax(
    ax,
    summary_df: pd.DataFrame,
    color_map: dict[str, str],
    value_col: str,
    ylabel: str,
    title: str,
    fmt: str,
    *,
    expand_acc_ylim: bool = False,
    yerr_col: str | None = None,
    significance_pairs: list[tuple[str, str, float]] | None = None,
) -> None:
    """
    yerr_col, when given, draws +/-1 sample-std error bars (a muted neutral color, not
    a series hue -- an error bar is a statistical annotation, not more data identity).
    significance_pairs, when given, draws a publication-style bracket + stars above
    each (method_a, method_b, pvalue) pair, stacked so multiple brackets don't collide.
    Both are no-ops when omitted -- every pre-existing (single-seed, untested) call
    site is unaffected.
    """
    labels = summary_df["method"].tolist()
    bar_colors = [color_map[m] for m in labels]
    values = summary_df[value_col].tolist()
    yerr = summary_df[yerr_col].tolist() if yerr_col is not None else None
    bars = ax.bar(
        range(len(labels)), values, color=bar_colors,
        yerr=yerr, capsize=4 if yerr is not None else 0,
        error_kw=({"ecolor": "#444444", "elinewidth": 1.2, "capthick": 1.2} if yerr is not None else None),
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9)
    if expand_acc_ylim:
        lo, hi = min(values), max(values)
        ax.set_ylim(lo - (hi - lo) * 0.5, hi + (hi - lo) * 0.2)
    if significance_pairs:
        tops = {lab: v + e for lab, v, e in zip(labels, values, yerr or [0.0] * len(values))}
        span = (max(values) - min(values)) or 1.0
        y_cursor = max(tops.values())
        for method_a, method_b, pvalue in significance_pairs:
            if method_a not in labels or method_b not in labels:
                continue
            x1, x2 = labels.index(method_a), labels.index(method_b)
            y_cursor += span * 0.08
            _draw_significance_bracket(ax, x1, x2, y_cursor, _significance_stars(pvalue))
        ax.set_ylim(top=y_cursor + span * 0.12)


def _aggregate_per_epoch(per_epoch_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Seed-averaged per-epoch accuracy: concatenates per-seed per_epoch_df frames and
    groups by (method, epoch). test_accuracy is the across-seed mean (reusing that
    exact column name so _draw_per_epoch_accuracy_ax's segment-detection logic works
    unchanged); test_accuracy_std is the across-seed sample std (ddof=1, pandas
    default -- consistent with this module's mean +/- std convention, NOT SEM or a
    CI), filled to 0.0 for the n=1-seed edge case (pandas .std() on one value is NaN;
    the module's own _std helper already treats <2 values as 0.0, matched here).
    model_type is taken via .first() per group, assuming it's consistent across seeds
    for the same (method, epoch) -- true here since every seed in one aggregate shares
    total_epochs / split_point / hyperparameters. train_time is also averaged (mean
    across seeds) so _add_cumulative_time can build a wall-clock x-axis on top of this
    result too, not just on a single seed's per_epoch_df.
    """
    combined = pd.concat(per_epoch_dfs, ignore_index=True)
    grouped = combined.groupby(["method", "epoch"], as_index=False).agg(
        test_accuracy=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        train_time=("train_time", "mean"),
        model_type=("model_type", "first"),
    )
    grouped["test_accuracy_std"] = grouped["test_accuracy_std"].fillna(0.0)
    return grouped


def _draw_weight_resolution_ablation_ax(ax, method_data: list[dict], title: str) -> None:
    """
    Grouped bar chart: one x-axis cluster per method, 3 bars each (baseline / continuous
    / int-rounded). Colored as an ordinal one-hue pair (continuous vs int-rounded, same
    hue at full vs half alpha) plus a neutral gray baseline reference -- not 3 unrelated
    categorical hues (see ABLATION_* constants). method_data: one dict per method, keys
    "method", "baseline", "continuous", "int_rounded", and optional "baseline_err" /
    "continuous_err" / "int_rounded_err" (None or absent -> no error bar for that bar,
    used for the per-seed chart where n=1 has no variance to show).
    """
    n = len(method_data)
    width = 0.25
    x = np.arange(n)
    fields = [
        ("baseline", "Baseline (native TM)", {"color": ABLATION_BASELINE_COLOR}),
        ("continuous", "Continuous (unrounded)", {"color": ABLATION_CONTINUOUS_COLOR}),
        ("int_rounded", "Int-rounded", {
            "color": ABLATION_CONTINUOUS_COLOR, "alpha": ABLATION_INT_ROUNDED_ALPHA,
            "edgecolor": ABLATION_CONTINUOUS_COLOR, "linewidth": 1.0,
        }),
    ]
    for i, (key, legend_label, style) in enumerate(fields):
        vals = [d[key] for d in method_data]
        errs = [d.get(f"{key}_err") for d in method_data]
        has_err = any(e is not None for e in errs)
        yerr = [e if e is not None else 0.0 for e in errs] if has_err else None
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset, vals, width, label=legend_label, yerr=yerr,
            capsize=4 if has_err else 0,
            error_kw=({"ecolor": "#444444", "elinewidth": 1.0, "capthick": 1.0} if has_err else None),
            **style,
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([d["method"] for d in method_data])
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, axis="y")
    # Legend at upper-right, with headroom added above the tallest bar+label so it
    # doesn't collide with the hyperparameter annotation box (bottom-right, see callers).
    ax.legend(fontsize=8, loc="upper right")
    # Truncate the bottom instead of starting at 0: baseline/continuous/int-rounded
    # all sit in a narrow band (e.g. 75-90%), and a 0-100 axis flattens exactly the
    # differences this chart exists to show. Same lo-span*0.5 / hi+span*k truncation
    # rule as _draw_bar_metric_ax's expand_acc_ylim, with extra top room for the legend.
    all_vals = [d[key] for d in method_data for key in ("baseline", "continuous", "int_rounded")]
    lo, hi = min(all_vals), max(all_vals)
    span = (hi - lo) or 1.0
    ax.set_ylim(lo - span * 0.5, hi + span * 0.6)


def write_result_plots_from_run_dir(run_dir: str) -> None:
    """
    Read per_epoch_results.csv + experiment_metadata.json; compute summary in memory; write seven PNGs.

    Raises:
        FileNotFoundError: if metadata or per-epoch CSV is missing.
    """
    meta_path = os.path.join(run_dir, EXPERIMENT_METADATA_FILENAME)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"Missing {EXPERIMENT_METADATA_FILENAME!r} under {run_dir!r}; required for plots."
        )
    per_epoch_path = os.path.join(run_dir, "per_epoch_results.csv")
    if not os.path.isfile(per_epoch_path):
        raise FileNotFoundError(f"Missing per_epoch_results.csv under {run_dir!r}.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    per_epoch_df = pd.read_csv(per_epoch_path)
    summary_df = compute_summary_df(per_epoch_df)
    split_one = int(meta["total_epochs"] * meta["split_point"])
    saved: list[str] = []

    with plt.rc_context(_SERIF_RCPARAMS):
        color_map = _plot_color_map()

        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_per_epoch_accuracy_ax(ax, per_epoch_df, color_map)
        _annotate_checkpoint(ax, split_one)
        _annotate_hyperparams_epoch_left_of_legend(ax, meta)
        p = os.path.join(run_dir, PLOT_PER_EPOCH_ACCURACY_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_PER_EPOCH_ACCURACY_PNG)

        # Wall-clock variant: an epoch is not a fixed unit of compute (a TM epoch
        # costs far more than an NN epoch), so this is the fair "who gets further for
        # equal compute" comparison -- no checkpoint marker here since the shared
        # checkpoint sits at a different cumulative time per method (different
        # epoch costs), unlike the epoch-indexed chart where it's one shared x value.
        time_df = _add_cumulative_time(per_epoch_df)
        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_per_epoch_accuracy_ax(ax, time_df, color_map, x_col="cumulative_train_time_s", x_label="Cumulative Training Time (s)")
        ax.set_title("Accuracy vs. Training Time")
        _annotate_hyperparams_epoch_left_of_legend(ax, meta)
        p = os.path.join(run_dir, PLOT_PER_EPOCH_ACCURACY_BY_TIME_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_PER_EPOCH_ACCURACY_BY_TIME_PNG)

        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_bar_metric_ax(
            ax, summary_df, color_map, "avg_last10_tm_accuracy",
            "Avg Last-10 TM Accuracy (%)", "Avg Last-10 TM Accuracy by Method", "{:.1f}%",
            expand_acc_ylim=True,
        )
        _annotate_hyperparams_bottom_right(ax, meta)
        p = os.path.join(run_dir, PLOT_AVG_LAST10_ACCURACY_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_AVG_LAST10_ACCURACY_PNG)

        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_bar_metric_ax(
            ax, summary_df, color_map, "avg_last10_tm_test_time_s",
            "Inference Time (s)", "Avg Last-10 TM Inference Time", "{:.3f}s",
        )
        _annotate_hyperparams_bottom_right(ax, meta)
        p = os.path.join(run_dir, PLOT_AVG_LAST10_TM_TEST_TIME_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_AVG_LAST10_TM_TEST_TIME_PNG)

        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_bar_metric_ax(
            ax, summary_df, color_map, "total_train_time_s",
            "Training Time (s)", "Total Training Time by Method", "{:.1f}s",
        )
        _annotate_hyperparams_bottom_right(ax, meta)
        p = os.path.join(run_dir, PLOT_TOTAL_TRAIN_TIME_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_TOTAL_TRAIN_TIME_PNG)

        # Weight-resolution ablation: load + build method_data here (before the
        # combined figure below, which includes this panel too), then also save it
        # as its own standalone PNG.
        method_data = None
        ablation_json_path = os.path.join(run_dir, WEIGHT_RESOLUTION_ABLATION_FILENAME)
        if os.path.isfile(ablation_json_path):
            with open(ablation_json_path, encoding="utf-8") as f:
                ablation = json.load(f)
            try:
                method_data = [
                    {
                        "method": m,
                        "baseline": ablation[f"{BASE_METHOD_OF[m]}_baseline_test_accuracy"],
                        "continuous": ablation[f"{m}_direct_continuous_test_accuracy"],
                        "int_rounded": ablation[f"{m}_direct_int_rounded_test_accuracy"],
                    }
                    for m in ("WTM-NN", "UTM-NN")
                ]
            except KeyError as e:
                logger.warning(f"Skipping {PLOT_WEIGHT_RESOLUTION_ABLATION_PNG}: ablation JSON at {ablation_json_path} missing key {e} (older schema).")
                method_data = None
            else:
                fig, ax = plt.subplots(figsize=(9, 6))
                _draw_weight_resolution_ablation_ax(ax, method_data, "Weight-Resolution Ablation")
                _annotate_hyperparams_bottom_right(ax, meta)
                p = os.path.join(run_dir, PLOT_WEIGHT_RESOLUTION_ABLATION_PNG)
                fig.savefig(p, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(PLOT_WEIGHT_RESOLUTION_ABLATION_PNG)
        else:
            logger.info(f"Skipping {PLOT_WEIGHT_RESOLUTION_ABLATION_PNG}: no {WEIGHT_RESOLUTION_ABLATION_FILENAME} found under {run_dir}.")

        fig, ((ax_curve, ax_curve_time, ax_ablation), (ax_bar, ax_time, ax_train)) = plt.subplots(2, 3, figsize=(21, 10))
        fig.suptitle(f"TM vs TM-NN — {meta['dataset_name']}", fontsize=13, fontweight="bold")
        _draw_per_epoch_accuracy_ax(ax_curve, per_epoch_df, color_map)
        _annotate_checkpoint(ax_curve, split_one)
        _annotate_hyperparams_epoch_left_of_legend(ax_curve, meta)
        _draw_per_epoch_accuracy_ax(ax_curve_time, time_df, color_map, x_col="cumulative_train_time_s", x_label="Cumulative Training Time (s)")
        ax_curve_time.set_title("Accuracy vs. Training Time")
        _annotate_hyperparams_epoch_left_of_legend(ax_curve_time, meta)
        if method_data is not None:
            _draw_weight_resolution_ablation_ax(ax_ablation, method_data, "Weight-Resolution Ablation")
            _annotate_hyperparams_bottom_right(ax_ablation, meta)
        else:
            ax_ablation.text(0.5, 0.5, "No ablation data available", ha="center", va="center", transform=ax_ablation.transAxes)
            ax_ablation.set_axis_off()
        _draw_bar_metric_ax(
            ax_bar, summary_df, color_map, "avg_last10_tm_accuracy",
            "Avg Last-10 TM Accuracy (%)", "Avg Last-10 TM Accuracy by Method", "{:.1f}%",
            expand_acc_ylim=True,
        )
        _annotate_hyperparams_bottom_right(ax_bar, meta)
        _draw_bar_metric_ax(
            ax_time, summary_df, color_map, "avg_last10_tm_test_time_s",
            "Inference Time (s)", "Avg Last-10 TM Inference Time", "{:.3f}s",
        )
        _annotate_hyperparams_bottom_right(ax_time, meta)
        _draw_bar_metric_ax(
            ax_train, summary_df, color_map, "total_train_time_s",
            "Training Time (s)", "Total Training Time by Method", "{:.1f}s",
        )
        _annotate_hyperparams_bottom_right(ax_train, meta)
        plt.tight_layout()
        p = os.path.join(run_dir, PLOT_COMBINED_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_COMBINED_PNG)

    logger.info(f"Saved {len(saved)} plots to {run_dir}: {', '.join(saved)}")


def write_aggregate_plots(
    aggregate_dir: str,
    per_epoch_dfs: list[pd.DataFrame],
    agg_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    ablation_agg_df: pd.DataFrame,
    *,
    dataset_name: str,
    C: int,
    T: int,
    s: float,
) -> None:
    """
    Cross-seed rollup plots for one dataset config, written once after the seed loop
    (see __main__). Six PNGs into aggregate_dir. Shaded bands / error bars everywhere
    show +/-1 across-seed sample std (ddof=1), NOT SEM or a confidence interval --
    consistent with this module's mean +/- std convention throughout.
    """
    meta = {"C": C, "T": T, "s": s}
    n_seeds = len(per_epoch_dfs)
    split_one = int(agg_df.iloc[0]["total_epochs"] * agg_df.iloc[0]["split_point"])
    saved: list[str] = []
    with plt.rc_context(_SERIF_RCPARAMS):
        color_map = _plot_color_map()

        # Seed-averaged per-epoch accuracy with +/-1 std band.
        agg_pe_df = _aggregate_per_epoch(per_epoch_dfs)
        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_per_epoch_accuracy_ax(ax, agg_pe_df, color_map, std_col="test_accuracy_std")
        _annotate_checkpoint(ax, split_one)
        ax.set_title(f"Per-Epoch Accuracy (mean +/- 1 std, n={n_seeds} seeds)")
        _annotate_hyperparams_epoch_left_of_legend(ax, meta)
        p = os.path.join(aggregate_dir, PLOT_AGGREGATE_PER_EPOCH_ACCURACY_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_AGGREGATE_PER_EPOCH_ACCURACY_PNG)

        # Wall-clock variant (see the per-seed docstring note in
        # write_result_plots_from_run_dir): no checkpoint marker here, since the
        # shared checkpoint sits at a different cumulative time per method.
        agg_time_df = _add_cumulative_time(agg_pe_df)
        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_per_epoch_accuracy_ax(
            ax, agg_time_df, color_map, std_col="test_accuracy_std",
            x_col="cumulative_train_time_s", x_label="Cumulative Training Time (s)",
        )
        ax.set_title(f"Accuracy vs. Training Time (mean +/- 1 std, n={n_seeds} seeds)")
        _annotate_hyperparams_epoch_left_of_legend(ax, meta)
        p = os.path.join(aggregate_dir, PLOT_AGGREGATE_PER_EPOCH_ACCURACY_BY_TIME_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_AGGREGATE_PER_EPOCH_ACCURACY_BY_TIME_PNG)

        # Aggregate accuracy bar chart, error bars + significance brackets.
        significance_pairs = [
            (row["treatment_method"], row["base_method"], row["ttest_pvalue"])
            for _, row in sig_df.iterrows()
        ]
        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_bar_metric_ax(
            ax, agg_df, color_map, "avg_last10_tm_accuracy_mean",
            "Avg Last-10 TM Accuracy (%)", f"Avg Last-10 TM Accuracy by Method (n={n_seeds} seeds)", "{:.1f}%",
            expand_acc_ylim=True, yerr_col="avg_last10_tm_accuracy_std", significance_pairs=significance_pairs,
        )
        _annotate_hyperparams_bottom_right(ax, meta)
        p = os.path.join(aggregate_dir, PLOT_AGGREGATE_ACCURACY_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_AGGREGATE_ACCURACY_PNG)

        # Aggregate inference-time and total-train-time, error bars, no brackets
        # (not statistically tested).
        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_bar_metric_ax(
            ax, agg_df, color_map, "avg_last10_tm_test_time_s_mean",
            "Inference Time (s)", "Avg Last-10 TM Inference Time (aggregate)", "{:.3f}s",
            yerr_col="avg_last10_tm_test_time_s_std",
        )
        _annotate_hyperparams_bottom_right(ax, meta)
        p = os.path.join(aggregate_dir, PLOT_AGGREGATE_TEST_TIME_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_AGGREGATE_TEST_TIME_PNG)

        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_bar_metric_ax(
            ax, agg_df, color_map, "total_train_time_s_mean",
            "Training Time (s)", "Total Training Time by Method (aggregate)", "{:.1f}s",
            yerr_col="total_train_time_s_std",
        )
        _annotate_hyperparams_bottom_right(ax, meta)
        p = os.path.join(aggregate_dir, PLOT_AGGREGATE_TRAIN_TIME_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(PLOT_AGGREGATE_TRAIN_TIME_PNG)

        # Aggregate weight-resolution ablation.
        method_data = []
        for m in ("WTM-NN", "UTM-NN"):
            matches = ablation_agg_df[ablation_agg_df["method"] == m]
            if matches.empty:
                continue
            row = matches.iloc[0]
            method_data.append({
                "method": m,
                "baseline": row["baseline_test_accuracy_mean"], "baseline_err": row["baseline_test_accuracy_std"],
                "continuous": row["direct_continuous_test_accuracy_mean"], "continuous_err": row["direct_continuous_test_accuracy_std"],
                "int_rounded": row["direct_int_rounded_test_accuracy_mean"], "int_rounded_err": row["direct_int_rounded_test_accuracy_std"],
            })
        if method_data:
            fig, ax = plt.subplots(figsize=(9, 6))
            _draw_weight_resolution_ablation_ax(ax, method_data, "Weight-Resolution Ablation (aggregate)")
            _annotate_hyperparams_bottom_right(ax, meta)
            p = os.path.join(aggregate_dir, PLOT_AGGREGATE_WEIGHT_RESOLUTION_PNG)
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(PLOT_AGGREGATE_WEIGHT_RESOLUTION_PNG)
        else:
            logger.info(f"Skipping {PLOT_AGGREGATE_WEIGHT_RESOLUTION_PNG}: no ablation rows available (0 seeds had ablation data).")

    logger.info(f"Saved {len(saved)} aggregate plots to {aggregate_dir}: {', '.join(saved)}")


def replot_from_run_dir(run_dir: str) -> None:
    """Alias for write_result_plots_from_run_dir (reads only per_epoch CSV + metadata; not summary CSV)."""
    write_result_plots_from_run_dir(run_dir)


def write_experiment_metadata(
    run_dir: str,
    *,
    C: int,
    T: int,
    s: float,
    dataset_name: str,
    number_of_state_bits: int,
    split_point: float,
    total_epochs: int,
    val_fraction: float,
) -> None:
    payload = {
        "schema_version": EXPERIMENT_METADATA_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "C": C,
        "T": T,
        "s": float(s),
        "dataset_name": dataset_name,
        "number_of_state_bits": number_of_state_bits,
        "split_point": float(split_point),
        "total_epochs": total_epochs,
        "val_fraction": float(val_fraction),
    }
    path = os.path.join(run_dir, EXPERIMENT_METADATA_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _config_dir(save_path: str, dataset_name: str, C: int, T: int, s: float, total_epochs: int) -> str:
    """Per-(dataset, hyperparameter-config) directory; parent of every seed_N/ and aggregate/ subdir."""
    return os.path.join(save_path, f"{dataset_name}_C{C}_T{T}_s{s}_e{total_epochs}")


def run_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    C: int,
    T: int,
    s: float,
    number_of_state_bits: int,
    total_epochs: int,
    save_path: str,
    seed_index: int, # 0-based; builds the seed_<seed_index> subdir. Distinct from the RNG
                      # seed value (BASE_SEED + i) passed separately to seed_everything.
    split_point: float = 0.5, # TM-phase fraction for WTM-NN / UTM-NN
    dataset_name: str = "MNIST", # plain dataset name -- seed identity lives in the directory, not here
    val_fraction: float = 0.1, # held out from x_train for weight scaling + best_theta selection
) -> Tuple[pd.DataFrame, pd.DataFrame, dict | None]:
    """
    Run experiment with given parameters. This will compare the following methods:
    - UTM: unweighted TM trained for total_epochs epochs (checkpoint pickled at split_one)
    - WTM: weighted TM trained for total_epochs epochs (checkpoint pickled at split_one)
    - WTM-NN: weighted TM trained for split_point*total_epochs epochs, then frozen; a
        differentiable weight head (LogWeightHead) is trained on its clause outputs for the rest
    - UTM-NN: same as WTM-NN but the base TM is unweighted

    NOTE on naming: "TM-NN" makes the direction of information flow explicit. Avoid "TM + NN" since
    it implies simultaneous use rather than sequential transfer.

    We will track:
    - Accuracy of each method on the test set at each epoch
    - Time per epoch for each method

    Returns:
        per_epoch_df: one row per training epoch with columns:
            method, epoch, model_type ("tm" | "nn"), test_accuracy, train_time, test_time
        summary_df: one row per method with columns:
            method, final_tm_accuracy, avg_tm_epoch_time_s
        ablation: dict from weight_resolution_ablation (fresh run), the same dict reloaded
            from weight_resolution_ablation.json (cache-hit skip, if that file exists), or
            None (cache-hit skip with no such file -- e.g. a run predating this field).

    Training metrics from train_tm / train_log_weight_head are list[EpochResult]: one dict per epoch
    with keys test_accuracy, test_time, train_time (see EpochResult near the top of this file).

    REPORTING ACCURACY:
    The TM is the deliverable (it is the interpretable model). Always report TM accuracy as the
    primary metric. NN accuracy is a secondary reference showing the ceiling the weights are pulled
    toward — show it as a footnote or secondary column, not the headline number.
    Final TM accuracy is always measured with scaled weights (scale_weights_for_tm applied first).

    PER-EPOCH ACCURACY CURVE:
    Two line styles on the same axis:
    - Solid line: model_type == "tm" (TM is actively being trained)
    - Dashed line: model_type == "nn" (NN head is being trained, TM frozen)

    RESULTS TABLE (suggested columns):
    | Method  | Avg Last-10 TM Acc | Time/Epoch (s) |
    |---------|--------------------|----------------|
    | UTM     |                    |                |
    | WTM     |                    |                |
    | UTM-NN  |                    |                |
    | WTM-NN  |                    |                |
    Pair the table with a bar chart (one bar per method, final TM accuracy) for scannability.
    The comparison isolates two axes: clause weighting (UTM vs WTM) and gradient-refined
    weights on frozen clause outputs (WTM-NN / UTM-NN vs their bases).

    After a successful training run, writes experiment_metadata.json next to the CSVs.
    Plots are written by write_result_plots_from_run_dir (five PNGs from per_epoch CSV + metadata).
    On skip (cached CSVs), plots are always regenerated; metadata must exist or FileNotFoundError is raised.
    """
    n_classes = len(np.unique(y_train))
    config_dir = _config_dir(save_path, dataset_name, C, T, s, total_epochs)
    run_dir = os.path.join(config_dir, f"{SEED_DIR_PREFIX}{seed_index}")
    per_epoch_csv = os.path.join(run_dir, "per_epoch_results.csv")
    summary_csv = os.path.join(run_dir, "summary_results.csv")
    metadata_json = os.path.join(run_dir, EXPERIMENT_METADATA_FILENAME)

    run_tag = f"{dataset_name} {SEED_DIR_PREFIX}{seed_index}"
    run_start = perf_counter()

    if os.path.isfile(per_epoch_csv) and os.path.isfile(summary_csv):
        logger.info(f"[{run_tag}] cached run found at {run_dir} -- skipping training, replotting only")
        if not os.path.isfile(metadata_json):
            raise FileNotFoundError(
                f"Missing {EXPERIMENT_METADATA_FILENAME!r} under {run_dir!r}; required to regenerate plots. "
                "Re-run training once for this configuration."
            )
        per_epoch_df = pd.read_csv(per_epoch_csv)
        summary_df = compute_summary_df(per_epoch_df)
        write_result_plots_from_run_dir(run_dir)
        ablation_json_path = os.path.join(run_dir, WEIGHT_RESOLUTION_ABLATION_FILENAME)
        if os.path.isfile(ablation_json_path):
            with open(ablation_json_path, encoding="utf-8") as f:
                ablation = json.load(f)
        else:
            ablation = None
        return per_epoch_df, summary_df, ablation

    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"[{run_tag}] starting: C={C} T={T} s={s} total_epochs={total_epochs} split_point={split_point}")

    # Shuffle once per run, then hold out a validation split from the training set.
    # The shuffle also gives each seed a distinct example order into the TM (the C
    # RNGs can't be seeded from Python - see seed_everything). All methods, TM
    # baselines included, train on the post-split training subset, so the comparison
    # stays apples-to-apples and val is genuinely unseen by every TM.
    perm = np.random.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    n_val = int(len(x_train) * val_fraction)
    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_train, y_train = x_train[n_val:], y_train[n_val:]

    split_one = int(total_epochs * split_point)
    split_two = total_epochs - split_one

    # Transient checkpoints only -- pass the split_one TM state from phase 1/2 into
    # phase 3/4 within this same run; deleted right after each is loaded, not a
    # persistent output artifact. No _C{C}_T{T}_s{s} suffix needed: run_dir already
    # sits under a config_dir that encodes them.
    unweighted_tm_path = os.path.join(run_dir, "unweighted_tm.pkl")
    weighted_tm_path = os.path.join(run_dir, "weighted_tm.pkl")

    rows: list[dict] = []

    def _add(method: str, epoch_results: list[EpochResult], model_type: str, start_epoch: int) -> int:
        for i, r in enumerate(epoch_results):
            rows.append({
                "method": method,
                "epoch": start_epoch + i,
                "model_type": model_type,
                "test_accuracy": r["test_accuracy"],
                "train_time": r["train_time"],
                "test_time": r["test_time"],
            })
        return start_epoch + len(epoch_results)

    def _tail10(epoch_results: list[EpochResult]) -> float:
        return float(np.mean([r["test_accuracy"] for r in epoch_results[-10:]]))

    # 1. UTM
    phase_start = perf_counter()
    logger.info(f"[{run_tag}] [1/4] Training UTM for {total_epochs} epochs")
    unweighted_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=False)
    r_u1 = train_tm(unweighted_tm, x_train, y_train, x_test, y_test, epochs=split_one)
    pkl.dump(unweighted_tm, open(unweighted_tm_path, "wb"))
    r_u2 = train_tm(unweighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    _add("UTM", r_u1 + r_u2, "tm", 1)
    logger.info(f"[{run_tag}] [1/4] UTM done in {perf_counter() - phase_start:.1f}s (avg-last10 acc {_tail10(r_u1 + r_u2):.2f}%)")

    # 2. WTM
    phase_start = perf_counter()
    logger.info(f"[{run_tag}] [2/4] Training WTM for {total_epochs} epochs")
    weighted_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=True)
    r_w1 = train_tm(weighted_tm, x_train, y_train, x_test, y_test, epochs=split_one)
    pkl.dump(weighted_tm, open(weighted_tm_path, "wb"))
    r_w2 = train_tm(weighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    _add("WTM", r_w1 + r_w2, "tm", 1)
    logger.info(f"[{run_tag}] [2/4] WTM done in {perf_counter() - phase_start:.1f}s (avg-last10 acc {_tail10(r_w1 + r_w2):.2f}%)")

    # 3. WTM-NN: reuse r_w1 as the TM phase (same checkpoint), then train NN
    phase_start = perf_counter()
    logger.info(f"[{run_tag}] [3/4] Training WTM-NN (NN phase, {split_two} epochs)")
    frozen_weighted_tm = pkl.load(open(weighted_tm_path, "rb"))
    os.remove(weighted_tm_path)  # done with the checkpoint file now that it's loaded
    nn_w = LogWeightHead(n_classes=n_classes, n_clauses=frozen_weighted_tm.number_of_clauses,
                         T=None, init_weights=frozen_weighted_tm.get_clause_weights())
    ep = _add("WTM-NN", r_w1, "tm", 1)
    r_nn_w = train_log_weight_head(nn_w, frozen_weighted_tm, x_train, y_train, x_val, y_val, x_test, y_test, epochs=split_two)
    ep = _add("WTM-NN", r_nn_w, "nn", ep)
    Z_val_w = frozen_weighted_tm.transform(x_val, inverted=False).astype("uint8")
    scaled_weights = scale_weights_for_tm(nn_w, Z_val_w, frozen_weighted_tm.T)
    frozen_weighted_tm.set_clause_weights(scaled_weights)
    logger.info(f"[{run_tag}] [3/4] WTM-NN done in {perf_counter() - phase_start:.1f}s (avg-last10 acc {_tail10(r_nn_w):.2f}%)")

    # 4. UTM-NN: reuse r_u1 as the TM phase, then train NN
    phase_start = perf_counter()
    logger.info(f"[{run_tag}] [4/4] Training UTM-NN (NN phase, {split_two} epochs)")
    frozen_unweighted_tm = pkl.load(open(unweighted_tm_path, "rb"))
    os.remove(unweighted_tm_path)  # done with the checkpoint file now that it's loaded
    nn_u = LogWeightHead(n_classes=n_classes, n_clauses=frozen_unweighted_tm.number_of_clauses, T=None)
    ep = _add("UTM-NN", r_u1, "tm", 1)
    r_nn_u = train_log_weight_head(nn_u, frozen_unweighted_tm, x_train, y_train, x_val, y_val, x_test, y_test, epochs=split_two)
    ep = _add("UTM-NN", r_nn_u, "nn", ep)
    Z_val_u = frozen_unweighted_tm.transform(x_val, inverted=False).astype("uint8")
    scaled_weights = scale_weights_for_tm(nn_u, Z_val_u, frozen_unweighted_tm.T)
    frozen_unweighted_tm.set_clause_weights(scaled_weights)
    logger.info(f"[{run_tag}] [4/4] UTM-NN done in {perf_counter() - phase_start:.1f}s (avg-last10 acc {_tail10(r_nn_u):.2f}%)")

    # Weight-resolution ablation: does the WTM-NN / UTM-NN gain survive collapsing
    # their gradient-fit real-valued weights onto the positive-integer, floor-1
    # lattice native weighted_clauses training is restricted to? Cheap (no
    # retraining) diagnostic for the continuous-vs-integer confound; not run on a
    # cache-hit skip since nn_w/nn_u aren't reconstructable from the cached CSVs.
    native_weights = weighted_tm.get_clause_weights()
    # per_epoch_df isn't built yet (see below) -- r_w1+r_w2 / r_u1+r_u2 are the same
    # raw epoch-result lists, in scope, matching compute_summary_df's tail-10 rule.
    wtm_baseline_acc = float(np.mean([r["test_accuracy"] for r in (r_w1 + r_w2)[-10:]]))
    utm_baseline_acc = float(np.mean([r["test_accuracy"] for r in (r_u1 + r_u2)[-10:]]))
    ablation = {
        "native_WTM_weight_min": float(native_weights.min()),
        "native_WTM_weight_max": float(native_weights.max()),
    }
    ablation.update(weight_resolution_ablation("WTM-NN", nn_w, frozen_weighted_tm, x_test, y_test, "WTM", wtm_baseline_acc))
    ablation.update(weight_resolution_ablation("UTM-NN", nn_u, frozen_unweighted_tm, x_test, y_test, "UTM", utm_baseline_acc))
    with open(os.path.join(run_dir, WEIGHT_RESOLUTION_ABLATION_FILENAME), "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)
    logger.info(f"[{run_tag}] Weight-resolution ablation (no clamp-avoidance rescale; rounding is the only difference within each pair):")
    for k, v in ablation.items():
        print(f"  {k}: {v}")

    per_epoch_df = pd.DataFrame(rows)

    # Inference time: single predict call on each final TM with scaled weights.
    # Scaling is a one-time training cost — deployed inference is just predict().
    final_models = {
        "UTM":    unweighted_tm,
        "WTM":      weighted_tm,
        "WTM-NN":   frozen_weighted_tm,
        "UTM-NN": frozen_unweighted_tm,
    }
    
    summary_df = compute_summary_df(per_epoch_df)

    per_epoch_df.to_csv(os.path.join(run_dir, "per_epoch_results.csv"), index=False)
    summary_df.to_csv(os.path.join(run_dir, "summary_results.csv"), index=False)
    logger.info(f"[{run_tag}] training complete in {perf_counter() - run_start:.1f}s")
    print(summary_df.to_string(index=False))

    write_experiment_metadata(
        run_dir,
        C=C,
        T=T,
        s=s,
        dataset_name=dataset_name,
        number_of_state_bits=number_of_state_bits,
        split_point=split_point,
        total_epochs=total_epochs,
        val_fraction=val_fraction,
    )
    write_result_plots_from_run_dir(run_dir)

    return per_epoch_df, summary_df, ablation


@dataclass
class CustomDataset:
    name: str
    train_dataset: Dataset
    test_dataset: Dataset
@dataclass
class ExperimentConfig:
    C: int
    T: int
    s: float
    number_of_state_bits: int
    total_epochs: int
    split_point: float
    dataset_name: str
    seed_index: int
    train_dataset: Dataset
    test_dataset: Dataset
    save_path: str
    val_fraction: float = 0.1
    def run(self):
        x_train, y_train, x_test, y_test = binarize_dataset(self.train_dataset, self.test_dataset)
        per_epoch_df, summary_df, ablation = run_experiment(
            x_train, y_train, x_test, y_test,
            C=self.C, T=self.T, s=self.s,
            number_of_state_bits=self.number_of_state_bits,
            total_epochs=self.total_epochs,
            split_point=self.split_point,
            dataset_name=self.dataset_name,
            seed_index=self.seed_index,
            save_path=self.save_path,
            val_fraction=self.val_fraction,
        )
        return per_epoch_df, summary_df, ablation


if __name__ == "__main__":
    """
    Findings:
    - at all clause levels, using the NN on top of the TM gives significant higher accuracy than the TM alone
    - low and high clause levels
    - Good results at C=50, T=12.5, s=4.0. Also with C=20, T=5, s=3.0.
    - Gains are higher at low clause levels, marginal at high clause levels


    Best per dataset:
    - EMNIST: C=300
    - FashionMNIST: C=100
    - KMNIST: C=100 for gain, C=1000 for accuracy
    - MNIST: C=100 for gain, C=500


    """

    C = 400
    T = C // 4
    s = 4.0
    number_of_state_bits = 8
    total_epochs = 500
    split_point = 0.3
    val_fraction = 0.1

    EMNISTDataset = CustomDataset(name="EMNIST", train_dataset=EMNIST(root="data", train=True, download=True, split="letters"), test_dataset=EMNIST(root="data", train=False, download=True, split="letters"))
    FashionMNISTDataset = CustomDataset(name="FashionMNIST", train_dataset=FashionMNIST(root="data", train=True, download=True), test_dataset=FashionMNIST(root="data", train=False, download=True))
    MNISTDataset = CustomDataset(name="MNIST", train_dataset=MNIST(root="data", train=True, download=True), test_dataset=MNIST(root="data", train=False, download=True))
    KMNISTDataset = CustomDataset(name="KMNIST", train_dataset=KMNIST(root="data", train=True, download=True), test_dataset=KMNIST(root="data", train=False, download=True))
    
    pairs = [
        (MNISTDataset, 100, 5),
        (FashionMNISTDataset, 100, 5),
        (KMNISTDataset, 100, 5),
        (EMNISTDataset, 300, 5),
    ]

    sweep_start = perf_counter()
    logger.info(f"Sweep starting: {len(pairs)} datasets, {sum(p[2] for p in pairs)} total runs")

    for dataset_idx, (dataset, C, seeds) in enumerate(pairs, 1):
        T = C // 4
        s = 4.0
        dataset_start = perf_counter()
        logger.info(f"=== Dataset {dataset_idx}/{len(pairs)}: {dataset.name} (C={C}, T={T}, s={s}, {seeds} seeds) ===")

        per_epoch_dfs = []
        summary_dfs = []
        ablations = []

        for i in range(seeds):
            seed_everything(BASE_SEED + i)
            logger.info(f"[{dataset.name} seed_{i}] seed {i+1}/{seeds} (rng seed={BASE_SEED + i})")
            config = ExperimentConfig(
                C=C, T=T, s=s,
                number_of_state_bits=number_of_state_bits,
                total_epochs=total_epochs,
                split_point=split_point,
                dataset_name=dataset.name,
                seed_index=i,
                train_dataset=dataset.train_dataset,
                test_dataset=dataset.test_dataset,
                save_path="results",
                val_fraction=val_fraction,
            )
            per_epoch_df, summary_df, ablation = config.run()
            per_epoch_dfs.append(per_epoch_df)
            summary_dfs.append(summary_df)
            ablations.append(ablation)

        config_dir = _config_dir("results", dataset.name, C, T, s, total_epochs)
        aggregate_dir = os.path.join(config_dir, AGGREGATE_DIRNAME)
        os.makedirs(aggregate_dir, exist_ok=True)

        agg_df = aggregate_experiment_results(
            dataset_name=dataset.name,
            per_epoch_dfs=per_epoch_dfs,
            summary_dfs=summary_dfs,
            save_path=aggregate_dir,
            total_epochs=total_epochs,
            T=T,
            s=s,
            split_point=split_point,
            val_fraction=val_fraction,
        )
        sig_df = aggregate_significance_tests(
            dataset_name=dataset.name,
            per_epoch_dfs=per_epoch_dfs,
            save_path=aggregate_dir,
            total_epochs=total_epochs,
            T=T,
            s=s,
        )
        ablation_agg_df = aggregate_weight_resolution_ablation(
            dataset_name=dataset.name,
            ablations=ablations,
            save_path=aggregate_dir,
            total_epochs=total_epochs,
            T=T,
            s=s,
        )
        write_aggregate_plots(
            aggregate_dir, per_epoch_dfs, agg_df, sig_df, ablation_agg_df,
            dataset_name=dataset.name, C=C, T=T, s=s,
        )
        logger.info(f"=== Dataset {dataset_idx}/{len(pairs)}: {dataset.name} complete in {(perf_counter() - dataset_start) / 60:.1f} min -> {aggregate_dir} ===")

    logger.info(f"Sweep complete: {len(pairs)} datasets in {(perf_counter() - sweep_start) / 60:.1f} min")