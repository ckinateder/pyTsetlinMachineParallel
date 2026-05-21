import json
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
from copy import deepcopy
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Per-epoch training metrics returned by train_tm / train_log_weight_head.
# Each list element is one epoch, in order. Keys are floats (seconds for times).
EpochResult = dict[str, float]
# Keys: "test_accuracy" (percent 0-100), "test_time" (s), "train_time" (s).

THRESHOLD = 75

# Summary table / aggregate order (also default row order in summary CSV).
METHODS_ORDERED = ("UTM", "WTM", "WTM-NN", "UTM-NN", "Cyclic")

# Per-epoch curve legend and color assignment (intentionally not METHODS_ORDERED).
METHODS_PLOT_ORDER = ("WTM-NN", "UTM-NN", "UTM", "WTM", "Cyclic")


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
            print("WARNING: Both init_weights and T are provided - this clamps gradients")
        
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


def train_log_weight_head(
    model: LogWeightHead,
    tm: MultiClassTsetlinMachine,
    x_train: np.ndarray,
    y_train: np.ndarray,
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

        # Report TM accuracy with current NN weights scaled and applied to eval_tm.
        test_start = perf_counter()
        scaled = scale_weights_for_tm(model, z_test, tm.T)
        eval_tm.set_clause_weights(scaled)
        acc = 100.0 * (eval_tm.predict(x_test) == y_test).mean()
        test_elapsed = perf_counter() - test_start

        results.append({"test_accuracy": acc, "test_time": test_elapsed, "train_time": train_elapsed})
        if acc > best_acc:
            best_acc = acc
            best_theta = model.theta.data.clone()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        pbar.set_postfix(tm_acc=f"{acc:.2f}%", loss=f"{loss.item():.2f}")

        if epochs_no_improve >= patience:
            break

    model.theta.data.copy_(best_theta)
    return results

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
) -> pd.DataFrame:
    """
    Per method: mean and sample std (ddof=1).
    - avg_last10_tm_accuracy: last 10 rows of test_accuracy per method/run (same rule as compute_summary_df);
      pool test_accuracy across runs - 10 * n_seeds values.
    - avg_tm_epoch_time_s, avg_last10_tm_test_time_s, total_train_time_s: recomputed / read like
      run_experiment summary (one scalar per run) - n_seeds values each.

    Optional kwargs total_epochs, T, s, split_point are copied into every output row (same
    experiment config for all seeds); omit or pass None to leave those CSV cells blank.
    """
    n = len(per_epoch_dfs)
    if n == 0:
        raise ValueError("aggregate_experiment_results: no runs")
    if len(summary_dfs) != n:
        raise ValueError("aggregate_experiment_results: per_epoch_dfs and summary_dfs length mismatch")

    def _std(vals: list[float]) -> float:
        if len(vals) < 2:
            return 0.0
        return float(np.std(vals, ddof=1))

    out_rows = []
    for method in METHODS_ORDERED:
        acc_vals: list[float] = []
        tm_train_vals: list[float] = []
        inf_vals: list[float] = []
        tot_vals: list[float] = []
        for pe_df, su_df in zip(per_epoch_dfs, summary_dfs):
            mdf = pe_df[pe_df["method"] == method]
            acc_vals.extend(mdf["test_accuracy"].tail(10).astype(float).tolist())
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
            "method": method,
            "n_seeds": n,
            "n_pooled_accuracy_epochs": len(acc_vals),
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
    out_path = os.path.join(save_path, f"{dataset_name}_aggregate_summary_results.csv")
    agg_df.to_csv(out_path, index=False)
    n_acc = out_rows[0]["n_pooled_accuracy_epochs"] if out_rows else 0
    print(f"\nAggregate summary ({n} seeds; accuracy pooled over {n_acc} per-epoch values) saved to {out_path}")
    print(agg_df.to_string(index=False))
    return agg_df


EXPERIMENT_METADATA_FILENAME = "experiment_metadata.json"
EXPERIMENT_METADATA_SCHEMA_VERSION = 1

# Five PNGs per run (no results.png): four singles + one 2x2 combined.
PLOT_PER_EPOCH_ACCURACY_PNG = "plot_per_epoch_accuracy.png"
PLOT_AVG_LAST10_ACCURACY_PNG = "plot_avg_last10_accuracy.png"
PLOT_AVG_LAST10_TM_TEST_TIME_PNG = "plot_avg_last10_tm_test_time.png"
PLOT_TOTAL_TRAIN_TIME_PNG = "plot_total_train_time.png"
PLOT_COMBINED_PNG = "plot_combined.png"

_SERIF_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
}


def _plot_color_map() -> dict[str, str]:
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {m: prop_cycle[i % len(prop_cycle)] for i, m in enumerate(METHODS_PLOT_ORDER)}


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


def _draw_per_epoch_accuracy_ax(ax, per_epoch_df: pd.DataFrame, color_map: dict[str, str]) -> None:
    max_epoch = per_epoch_df["epoch"].max()
    for method in METHODS_PLOT_ORDER:
        color = color_map[method]
        train_df = per_epoch_df[per_epoch_df["method"] == method].sort_values("epoch").reset_index(drop=True)
        if train_df.empty:
            continue
        train_df = train_df.copy()
        train_df["segment"] = (train_df["model_type"] != train_df["model_type"].shift()).cumsum()
        first = True
        for seg_id, seg in train_df.groupby("segment"):
            ls = "--" if seg["model_type"].iloc[0] == "nn" else "-"
            if seg_id > train_df["segment"].min():
                prev = train_df[train_df["segment"] == seg_id - 1].iloc[-1]
                epochs = [prev["epoch"]] + list(seg["epoch"])
                accs = [prev["test_accuracy"]] + list(seg["test_accuracy"])
            else:
                epochs = list(seg["epoch"])
                accs = list(seg["test_accuracy"])
            ax.plot(epochs, accs, color=color, linestyle=ls, linewidth=1.5, label=method if first else "_nolegend_")
            first = False
        last = train_df.iloc[-1]
        if last["epoch"] < max_epoch:
            ax.plot(
                [last["epoch"], max_epoch], [last["test_accuracy"]] * 2,
                color=color, linestyle=":", linewidth=1.0, alpha=0.4,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Per-Epoch Accuracy")
    ax.grid(True, alpha=0.3, zorder=0)
    method_handles = [Line2D([0], [0], color=color_map[m], linewidth=1.5, label=m) for m in METHODS_PLOT_ORDER]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="TM training"),
        Line2D([0], [0], color="black", linestyle="--", label="NN training"),
    ]
    ax.legend(handles=method_handles + style_handles, fontsize=8, loc="lower right")


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
) -> None:
    labels = summary_df["method"].tolist()
    bar_colors = [color_map[m] for m in labels]
    values = summary_df[value_col].tolist()
    bars = ax.bar(range(len(labels)), values, color=bar_colors)
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


def write_result_plots_from_run_dir(run_dir: str) -> None:
    """
    Read per_epoch_results.csv + experiment_metadata.json; compute summary in memory; write five PNGs.

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

    with plt.rc_context(_SERIF_RCPARAMS):
        color_map = _plot_color_map()

        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_per_epoch_accuracy_ax(ax, per_epoch_df, color_map)
        _annotate_hyperparams_epoch_left_of_legend(ax, meta)
        p = os.path.join(run_dir, PLOT_PER_EPOCH_ACCURACY_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {p}")

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
        print(f"Saved plot to {p}")

        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_bar_metric_ax(
            ax, summary_df, color_map, "avg_last10_tm_test_time_s",
            "Inference Time (s)", "Avg Last-10 TM Inference Time", "{:.3f}s",
        )
        _annotate_hyperparams_bottom_right(ax, meta)
        p = os.path.join(run_dir, PLOT_AVG_LAST10_TM_TEST_TIME_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {p}")

        fig, ax = plt.subplots(figsize=(9, 6))
        _draw_bar_metric_ax(
            ax, summary_df, color_map, "total_train_time_s",
            "Training Time (s)", "Total Training Time by Method", "{:.1f}s",
        )
        _annotate_hyperparams_bottom_right(ax, meta)
        p = os.path.join(run_dir, PLOT_TOTAL_TRAIN_TIME_PNG)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {p}")

        fig, ((ax_curve, ax_time), (ax_bar, ax_train)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"TM vs TM⟷NN — {meta['dataset_name']}", fontsize=13, fontweight="bold")
        _draw_per_epoch_accuracy_ax(ax_curve, per_epoch_df, color_map)
        _annotate_hyperparams_epoch_left_of_legend(ax_curve, meta)
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
        print(f"Saved plot to {p}")


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
    rounds: int,
    epochs_per_round: int,
    split_point: float,
    total_epochs: int,
) -> None:
    payload = {
        "schema_version": EXPERIMENT_METADATA_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "C": C,
        "T": T,
        "s": float(s),
        "dataset_name": dataset_name,
        "number_of_state_bits": number_of_state_bits,
        "rounds": rounds,
        "epochs_per_round": epochs_per_round,
        "split_point": float(split_point),
        "total_epochs": total_epochs,
    }
    path = os.path.join(run_dir, EXPERIMENT_METADATA_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_experiment(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    C: int,
    T: int,
    s: float,
    number_of_state_bits: int,
    rounds: int,
    epochs_per_round: int,
    save_path: str,
    split_point: float = 0.5, # for weighted TM-NN and unweighted TM-NN
    dataset_name: str = "MNIST",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run experiment with given parameters. This will compare the following methods:
    - UTM: unweighted TM for rounds*epochs_per_round epochs (save to pickle halfway)
    - WTM: weighted TM for rounds*epochs_per_round epochs (save to pickle halfway)
    - WTM-NN: frozen weighted TM trained for rounds*epochs_per_round / 2 epochs, then NN
        trained on top for rounds*epochs_per_round / 2 epochs
    - UTM-NN: frozen unweighted TM trained for rounds*epochs_per_round / 2 epochs, then
        NN trained on top for rounds*epochs_per_round / 2 epochs
    - Cyclic (TM⟷NN): start with weighted TM, alternate between TM and NN every
        epochs_per_round epochs. Each round: NN is initialized from the TM's current clause weights,
        trained for epochs_per_round epochs, then weights are copied back to the TM.

    NOTE on naming: "TM-NN" makes the direction of information flow explicit. Avoid "TM + NN" since
    it implies simultaneous use rather than sequential transfer.

    We will track:
    - Accuracy of each method on the test set at each epoch
    - Time per epoch for each method

    Returns:
        per_epoch_df: one row per training epoch with columns:
            method, epoch, model_type ("tm" | "nn" | "tm_sync"), test_accuracy,
            train_time, test_time, is_sync
        summary_df: one row per method with columns:
            method, final_tm_accuracy, avg_tm_epoch_time_s

    Training metrics from train_tm / train_log_weight_head are list[EpochResult]: one dict per epoch
    with keys test_accuracy, test_time, train_time (see EpochResult near the top of this file).

    REPORTING ACCURACY:
    The TM is the deliverable (it is the interpretable model). Always report TM accuracy as the
    primary metric. NN accuracy is a secondary reference showing the ceiling the weights are pulled
    toward — show it as a footnote or secondary column, not the headline number.
    Final TM accuracy is always measured with scaled weights (scale_weights_for_tm applied first).

    PER-EPOCH ACCURACY CURVE (cyclic method):
    Use two line styles on the same axis:
    - Solid line: model_type == "tm" (TM is actively being trained)
    - Dashed line: model_type == "nn" (NN is being trained, TM frozen)
    - is_sync == True marks where NN weights are scaled and copied back to the TM
    The TM accuracy stays flat during dashed segments and jumps at each sync point, producing a
    staircase shape that visually communicates the cyclic mechanic. Add shaded regions or rug
    ticks at sync points to make the jumps legible.

    RESULTS TABLE (suggested columns):
    | Method            | Avg Last-10 TM Acc               | Time/Epoch (s) |
    |-------------------|----------------------------------|----------------|
    | UTM     |                                  |                |
    | WTM       |                                  |                |
    | UTM-NN  |                                  |                |
    | WTM-NN    |                                  |                |
    | Cyclic (ours)|                                  |                |
    Pair the table with a bar chart (one bar per method, final TM accuracy) for scannability.
    The central claim: cyclic wins on accuracy; the per-epoch curve shows why (continuous
    bidirectional refinement vs. one-shot transfer).

    After a successful training run, writes experiment_metadata.json next to the CSVs.
    Plots are written by write_result_plots_from_run_dir (five PNGs from per_epoch CSV + metadata).
    On skip (cached CSVs), plots are always regenerated; metadata must exist or FileNotFoundError is raised.
    """
    n_classes = len(np.unique(y_train))
    total_epochs = rounds * epochs_per_round * 2
    run_dir = os.path.join(save_path, f"{dataset_name}_C{C}_T{T}_s{s}_e{total_epochs}")
    per_epoch_csv = os.path.join(run_dir, "per_epoch_results.csv")
    summary_csv = os.path.join(run_dir, "summary_results.csv")
    metadata_json = os.path.join(run_dir, EXPERIMENT_METADATA_FILENAME)

    if os.path.isfile(per_epoch_csv) and os.path.isfile(summary_csv):
        print(f"Run directory {run_dir} already exists and CSV results are present. Skipping training...")
        if not os.path.isfile(metadata_json):
            raise FileNotFoundError(
                f"Missing {EXPERIMENT_METADATA_FILENAME!r} under {run_dir!r}; required to regenerate plots. "
                "Re-run training once for this configuration."
            )
        per_epoch_df = pd.read_csv(per_epoch_csv)
        summary_df = compute_summary_df(per_epoch_df)
        write_result_plots_from_run_dir(run_dir)
        return per_epoch_df, summary_df

    os.makedirs(run_dir, exist_ok=True)
    split_one = int(total_epochs * split_point)
    split_two = total_epochs - split_one

    unweighted_tm_path = os.path.join(run_dir, f"unweighted_tm_C{C}_T{T}_s{s}.pkl")
    weighted_tm_path = os.path.join(run_dir, f"weighted_tm_C{C}_T{T}_s{s}.pkl")

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

    # 5. Cyclic
    print(f"[5/5] Cyclic TM⟷NN ({rounds} rounds × {epochs_per_round} epochs each phase)")
    cyclic_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=True)
    alt_epoch = 1
    pbar = tqdm(range(rounds), desc="Cyclic", dynamic_ncols=True, leave=False)
    for rnd in pbar:
        r_tm = train_tm(cyclic_tm, x_train, y_train, x_test, y_test, epochs=epochs_per_round)
        alt_epoch = _add("Cyclic", r_tm, "tm", alt_epoch)

        nn_a = LogWeightHead(n_classes=n_classes, n_clauses=cyclic_tm.number_of_clauses,
                             T=None, init_weights=cyclic_tm.get_clause_weights())
        r_nn = train_log_weight_head(nn_a, cyclic_tm, x_train, y_train, x_test, y_test, epochs=epochs_per_round, patience=10)
        alt_epoch = _add("Cyclic", r_nn, "nn", alt_epoch)

        Z_test_a = cyclic_tm.transform(x_test, inverted=False).astype("uint8")
        scaled_weights = scale_weights_for_tm(nn_a, Z_test_a, cyclic_tm.T)
        cyclic_tm.set_clause_weights(scaled_weights)

        # copy unscaled weights back so next TM round has a warm start
        cyclic_tm.set_clause_weights(nn_a.weights.detach().cpu().numpy())
    pbar.close()

    # 1. UTM
    print(f"[1/5] Training UTM for {total_epochs} epochs")
    unweighted_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=False)
    r_u1 = train_tm(unweighted_tm, x_train, y_train, x_test, y_test, epochs=split_one)
    pkl.dump(unweighted_tm, open(unweighted_tm_path, "wb"))
    r_u2 = train_tm(unweighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    _add("UTM", r_u1 + r_u2, "tm", 1)

    # 2. WTM
    print(f"[2/5] Training WTM for {total_epochs} epochs")
    weighted_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=True)
    r_w1 = train_tm(weighted_tm, x_train, y_train, x_test, y_test, epochs=split_one)
    pkl.dump(weighted_tm, open(weighted_tm_path, "wb"))
    r_w2 = train_tm(weighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    _add("WTM", r_w1 + r_w2, "tm", 1)

    # 3. WTM-NN: reuse r_w1 as the TM phase (same checkpoint), then train NN
    print(f"[3/5] Training WTM-NN (NN phase, {split_two} epochs)")
    frozen_weighted_tm = pkl.load(open(weighted_tm_path, "rb"))
    nn_w = LogWeightHead(n_classes=n_classes, n_clauses=frozen_weighted_tm.number_of_clauses,
                         T=None, init_weights=frozen_weighted_tm.get_clause_weights())
    ep = _add("WTM-NN", r_w1, "tm", 1)
    r_nn_w = train_log_weight_head(nn_w, frozen_weighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    ep = _add("WTM-NN", r_nn_w, "nn", ep)
    Z_test_w = frozen_weighted_tm.transform(x_test, inverted=False).astype("uint8")
    scaled_weights = scale_weights_for_tm(nn_w, Z_test_w, frozen_weighted_tm.T)
    frozen_weighted_tm.set_clause_weights(scaled_weights)

    # 4. UTM-NN: reuse r_u1 as the TM phase, then train NN
    print(f"[4/5] Training UTM-NN (NN phase, {split_two} epochs)")
    frozen_unweighted_tm = pkl.load(open(unweighted_tm_path, "rb"))
    nn_u = LogWeightHead(n_classes=n_classes, n_clauses=frozen_unweighted_tm.number_of_clauses, T=None)
    ep = _add("UTM-NN", r_u1, "tm", 1)
    r_nn_u = train_log_weight_head(nn_u, frozen_unweighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    ep = _add("UTM-NN", r_nn_u, "nn", ep)
    Z_test_u = frozen_unweighted_tm.transform(x_test, inverted=False).astype("uint8")
    scaled_weights = scale_weights_for_tm(nn_u, Z_test_u, frozen_unweighted_tm.T)
    frozen_unweighted_tm.set_clause_weights(scaled_weights)

    per_epoch_df = pd.DataFrame(rows)

    # Inference time: single predict call on each final TM with scaled weights.
    # Scaling is a one-time training cost — deployed inference is just predict().
    if rounds > 0:
        Z_test_a = cyclic_tm.transform(x_test, inverted=False).astype("uint8")
        cyclic_tm.set_clause_weights(scale_weights_for_tm(nn_a, Z_test_a, cyclic_tm.T))
    final_models = {
        "UTM":    unweighted_tm,
        "WTM":      weighted_tm,
        "WTM-NN":   frozen_weighted_tm,
        "UTM-NN": frozen_unweighted_tm,
        "Cyclic":      cyclic_tm,
    }
    
    summary_df = compute_summary_df(per_epoch_df)

    per_epoch_df.to_csv(os.path.join(run_dir, "per_epoch_results.csv"), index=False)
    summary_df.to_csv(os.path.join(run_dir, "summary_results.csv"), index=False)
    print("\n" + summary_df.to_string(index=False))

    write_experiment_metadata(
        run_dir,
        C=C,
        T=T,
        s=s,
        dataset_name=dataset_name,
        number_of_state_bits=number_of_state_bits,
        rounds=rounds,
        epochs_per_round=epochs_per_round,
        split_point=split_point,
        total_epochs=total_epochs,
    )
    write_result_plots_from_run_dir(run_dir)

    return per_epoch_df, summary_df


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
    rounds: int
    epochs_per_round: int
    split_point: float
    dataset_name: str
    train_dataset: Dataset
    test_dataset: Dataset
    save_path: str
    def run(self):
        x_train, y_train, x_test, y_test = binarize_dataset(self.train_dataset, self.test_dataset)
        per_epoch_df, summary_df = run_experiment(
            x_train, y_train, x_test, y_test,
            C=self.C, T=self.T, s=self.s,
            number_of_state_bits=self.number_of_state_bits,
            rounds=self.rounds,
            epochs_per_round=self.epochs_per_round,
            split_point=self.split_point,
            dataset_name=self.dataset_name,
            save_path=self.save_path,
        )
        return per_epoch_df, summary_df


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    """
    Findings:
    - at all clause levels, using the NN on top of the TM gives significant higher accuracy than the TM alone
    - low and high clause levels
    - cyclic TM/NN is better than frozen TM at base
    - Good results at C=50, T=12.5, s=4.0, 15 rounds of 15. Also with C=20, T=5, s=3.0, 15 rounds of 15.
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
    rounds = 5
    epochs_per_round = 50
    split_point = 0.3

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

    for dataset, C, seeds in pairs:
        T = C // 4
        s = 4.0

        per_epoch_dfs = []
        summary_dfs = []

        for i in range(seeds):
            print(f"Running experiment for {dataset.name} with C={C}, T={T}, s={s}, seed {i+1} of {seeds}")
            config = ExperimentConfig(
                C=C, T=T, s=s,
                rounds=rounds,
                number_of_state_bits=number_of_state_bits,
                epochs_per_round=epochs_per_round,
                split_point=split_point,
                dataset_name=f"{dataset.name}_r{i}",
                train_dataset=dataset.train_dataset,
                test_dataset=dataset.test_dataset,
                save_path="results",
            )
            per_epoch_df, summary_df = config.run()
            per_epoch_dfs.append(per_epoch_df)
            summary_dfs.append(summary_df)

        aggregate_experiment_results(
            dataset_name=dataset.name,
            per_epoch_dfs=per_epoch_dfs,
            summary_dfs=summary_dfs,
            save_path="results",
            total_epochs=rounds * epochs_per_round * 2,
            T=T,
            s=s,
            split_point=split_point,
        )


    """
    datasets = [EMNISTDataset, FashionMNISTDataset, MNISTDataset, KMNISTDataset]
    #random.shuffle(datasets)
    C_values = [25, 50, 100, 300, 500, 1000]
    permutes = list(itertools.product(datasets, C_values))
    random.shuffle(permutes)

    for dataset, C in permutes:
        T = C // 4
        print(f"Running experiment for {dataset.name} with C={C}, T={T}, s={s}")
        config = ExperimentConfig(
            C=C, T=T, s=s,
            number_of_state_bits=number_of_state_bits,
            rounds=rounds,
            epochs_per_round=epochs_per_round,
            split_point=split_point,
            dataset_name=dataset.name,
            train_dataset=dataset.train_dataset,
            test_dataset=dataset.test_dataset,
            save_path="results",
        )
        config.run()
    """