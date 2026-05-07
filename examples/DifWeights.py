import copy
import random
from time import perf_counter
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from torchvision.datasets import MNIST, EMNIST, KMNIST, FashionMNIST
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import os
from copy import deepcopy

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

@dataclass
class TMConfig:
    C: int
    T: int
    s: float
    weighted_clauses: bool
    number_of_state_bits: int

# Per-epoch training metrics returned by train_tm / train_log_weight_head.
# Each list element is one epoch, in order. Keys are floats (seconds for times).
EpochResult = dict[str, float]
# Keys: "test_accuracy" (percent 0-100), "test_time" (s), "train_time" (s).

THRESHOLD = 75

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_weight)

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
            loss = F.cross_entropy(model(zb), yb)
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
    split_point: float = 0.5, # for weighted TM→NN and unweighted TM→NN
    dataset_name: str = "MNIST",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run experiment with given parameters. This will compare the following methods:
    - TM-Unweighted: unweighted TM for rounds*epochs_per_round epochs (save to pickle halfway)
    - TM-Weighted: weighted TM for rounds*epochs_per_round epochs (save to pickle halfway)
    - TM-Weighted→NN: frozen weighted TM trained for rounds*epochs_per_round / 2 epochs, then NN
        trained on top for rounds*epochs_per_round / 2 epochs
    - TM-Unweighted→NN: frozen unweighted TM trained for rounds*epochs_per_round / 2 epochs, then
        NN trained on top for rounds*epochs_per_round / 2 epochs
    - Alternating (TM⟷NN): start with weighted TM, alternate between TM and NN every
        epochs_per_round epochs. Each round: NN is initialized from the TM's current clause weights,
        trained for epochs_per_round epochs, then weights are copied back to the TM.

    NOTE on naming: "TM→NN" makes the direction of information flow explicit. Avoid "TM + NN" since
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

    PER-EPOCH ACCURACY CURVE (alternating method):
    Use two line styles on the same axis:
    - Solid line: model_type == "tm" (TM is actively being trained)
    - Dashed line: model_type == "nn" (NN is being trained, TM frozen)
    - is_sync == True marks where NN weights are scaled and copied back to the TM
    The TM accuracy stays flat during dashed segments and jumps at each sync point, producing a
    staircase shape that visually communicates the alternating mechanic. Add shaded regions or rug
    ticks at sync points to make the jumps legible.

    RESULTS TABLE (suggested columns):
    | Method            | Final TM Acc (avg last k epochs) | Time/Epoch (s) |
    |-------------------|----------------------------------|----------------|
    | TM-Unweighted     |                                  |                |
    | TM-Weighted       |                                  |                |
    | TM-Unweighted→NN  |                                  |                |
    | TM-Weighted→NN    |                                  |                |
    | Alternating (ours)|                                  |                |
    Pair the table with a bar chart (one bar per method, final TM accuracy) for scannability.
    The central claim: alternating wins on accuracy; the per-epoch curve shows why (continuous
    bidirectional refinement vs. one-shot transfer).
    """
    run_dir = os.path.join(save_path, f"{dataset_name}_C{C}_T{T}_s{s}")
    os.makedirs(run_dir, exist_ok=True)
    n_classes = len(np.unique(y_train))
    total_epochs = rounds * epochs_per_round * 2
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

# 5. Alternating
    print(f"[5/5] Alternating TM⟷NN ({rounds} rounds × {epochs_per_round} epochs each phase)")
    alternating_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=True)
    alt_epoch = 1
    pbar = tqdm(range(rounds), desc="Alternating", dynamic_ncols=True, leave=False)
    for rnd in pbar:
        r_tm = train_tm(alternating_tm, x_train, y_train, x_test, y_test, epochs=epochs_per_round)
        alt_epoch = _add("Alternating", r_tm, "tm", alt_epoch)

        nn_a = LogWeightHead(n_classes=n_classes, n_clauses=alternating_tm.number_of_clauses,
                             T=None, init_weights=alternating_tm.get_clause_weights())
        r_nn = train_log_weight_head(nn_a, alternating_tm, x_train, y_train, x_test, y_test, epochs=epochs_per_round)
        alt_epoch = _add("Alternating", r_nn, "nn", alt_epoch)

        Z_test_a = alternating_tm.transform(x_test, inverted=False).astype("uint8")
        scaled_weights = scale_weights_for_tm(nn_a, Z_test_a, alternating_tm.T)
        alternating_tm.set_clause_weights(scaled_weights)

        # copy unscaled weights back so next TM round has a warm start
        alternating_tm.set_clause_weights(nn_a.weights.detach().cpu().numpy())
    pbar.close()

    # 1. TM-Unweighted
    print(f"[1/5] Training TM-Unweighted for {total_epochs} epochs")
    unweighted_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=False)
    r_u1 = train_tm(unweighted_tm, x_train, y_train, x_test, y_test, epochs=split_one)
    pkl.dump(unweighted_tm, open(unweighted_tm_path, "wb"))
    r_u2 = train_tm(unweighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    _add("TM-Unweighted", r_u1 + r_u2, "tm", 1)

    # 2. TM-Weighted
    print(f"[2/5] Training TM-Weighted for {total_epochs} epochs")
    weighted_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=True)
    r_w1 = train_tm(weighted_tm, x_train, y_train, x_test, y_test, epochs=split_one)
    pkl.dump(weighted_tm, open(weighted_tm_path, "wb"))
    r_w2 = train_tm(weighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    _add("TM-Weighted", r_w1 + r_w2, "tm", 1)

    # 3. TM-Weighted→NN: reuse r_w1 as the TM phase (same checkpoint), then train NN
    print(f"[3/5] Training TM-Weighted→NN (NN phase, {split_two} epochs)")
    frozen_weighted_tm = pkl.load(open(weighted_tm_path, "rb"))
    nn_w = LogWeightHead(n_classes=n_classes, n_clauses=frozen_weighted_tm.number_of_clauses,
                         T=None, init_weights=frozen_weighted_tm.get_clause_weights())
    ep = _add("TM-Weighted→NN", r_w1, "tm", 1)
    r_nn_w = train_log_weight_head(nn_w, frozen_weighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    ep = _add("TM-Weighted→NN", r_nn_w, "nn", ep)
    Z_test_w = frozen_weighted_tm.transform(x_test, inverted=False).astype("uint8")
    scaled_weights = scale_weights_for_tm(nn_w, Z_test_w, frozen_weighted_tm.T)
    frozen_weighted_tm.set_clause_weights(scaled_weights)

    # 4. TM-Unweighted→NN: reuse r_u1 as the TM phase, then train NN
    print(f"[4/5] Training TM-Unweighted→NN (NN phase, {split_two} epochs)")
    frozen_unweighted_tm = pkl.load(open(unweighted_tm_path, "rb"))
    nn_u = LogWeightHead(n_classes=n_classes, n_clauses=frozen_unweighted_tm.number_of_clauses, T=None)
    ep = _add("TM-Unweighted→NN", r_u1, "tm", 1)
    r_nn_u = train_log_weight_head(nn_u, frozen_unweighted_tm, x_train, y_train, x_test, y_test, epochs=split_two)
    ep = _add("TM-Unweighted→NN", r_nn_u, "nn", ep)
    Z_test_u = frozen_unweighted_tm.transform(x_test, inverted=False).astype("uint8")
    scaled_weights = scale_weights_for_tm(nn_u, Z_test_u, frozen_unweighted_tm.T)
    frozen_unweighted_tm.set_clause_weights(scaled_weights)

    per_epoch_df = pd.DataFrame(rows)

    # Summary: read final accuracy from per_epoch_df so bar chart matches curve endpoints exactly.
    methods_ordered = ["TM-Unweighted", "TM-Weighted", "TM-Weighted→NN", "TM-Unweighted→NN", "Alternating"]
    summary_rows = []
    for method in methods_ordered:
        mdf = per_epoch_df[per_epoch_df["method"] == method]
        final_tm_acc = mdf["test_accuracy"].max()
        avg_tm_time = mdf[mdf["model_type"] == "tm"]["train_time"].mean()
        summary_rows.append({
            "method": method,
            "best_tm_accuracy": round(final_tm_acc, 2),
            "avg_tm_epoch_time_s": round(avg_tm_time, 3),
        })
    summary_df = pd.DataFrame(summary_rows)

    per_epoch_df.to_csv(os.path.join(run_dir, "per_epoch_results.csv"), index=False)
    summary_df.to_csv(os.path.join(run_dir, "summary_results.csv"), index=False)
    print("\n" + summary_df.to_string(index=False))

    plot_results(per_epoch_df, summary_df, run_dir, C=C, T=T, s=s, dataset_name=dataset_name)

    return per_epoch_df, summary_df


def plot_results(
    per_epoch_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    save_path: str,
    C: int = None,
    T: int = None,
    s: float = None,
    dataset_name: str = "MNIST",
) -> None:
    methods = ["TM-Unweighted", "TM-Weighted", "TM-Weighted→NN", "TM-Unweighted→NN", "Alternating"]
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {m: prop_cycle[i % len(prop_cycle)] for i, m in enumerate(methods)}

    fig, (ax_curve, ax_bar) = plt.subplots(2, 1, figsize=(10, 10))

    fig.suptitle(f"TM vs TM⟷NN — {dataset_name}", fontsize=13, fontweight="bold")

    # --- per-epoch accuracy curve ---
    for method in methods:
        color = color_map[method]
        train_df = per_epoch_df[per_epoch_df["method"] == method].sort_values("epoch").reset_index(drop=True)
        if train_df.empty:
            continue

        # group consecutive rows of the same model_type into segments
        train_df["segment"] = (train_df["model_type"] != train_df["model_type"].shift()).cumsum()
        first = True
        for seg_id, seg in train_df.groupby("segment"):
            ls = "--" if seg["model_type"].iloc[0] == "nn" else "-"
            # prepend last point of previous segment to keep the line continuous
            if seg_id > train_df["segment"].min():
                prev = train_df[train_df["segment"] == seg_id - 1].iloc[-1]
                epochs = [prev["epoch"]] + list(seg["epoch"])
                accs   = [prev["test_accuracy"]] + list(seg["test_accuracy"])
            else:
                epochs = list(seg["epoch"])
                accs   = list(seg["test_accuracy"])
            ax_curve.plot(epochs, accs, color=color, linestyle=ls, linewidth=1.5,
                          label=method if first else "_nolegend_")
            first = False

    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Test Accuracy (%)")
    ax_curve.set_title("Per-Epoch Accuracy")
    ax_curve.grid(True, alpha=0.3)

    # param box (upper-left)
    if C is not None:
        param_text = f"C = {C}\nT = {T}\ns = {s}"
        ax_curve.text(0.02, 0.97, param_text, transform=ax_curve.transAxes,
                      fontsize=9, verticalalignment="top",
                      bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.8))

    # legend: method colors + line-style key
    method_handles = [Line2D([0], [0], color=color_map[m], linewidth=1.5, label=m) for m in methods]
    style_handles  = [Line2D([0], [0], color="black", linestyle="-",  label="TM training"),
                      Line2D([0], [0], color="black", linestyle="--", label="NN training")]
    ax_curve.legend(handles=method_handles + style_handles, fontsize=8, loc="lower right")

    # --- final accuracy bar chart ---
    labels     = summary_df["method"].tolist()
    accuracies = summary_df["best_tm_accuracy"].tolist()
    bar_colors = [color_map[m] for m in labels]
    bars = ax_bar.bar(range(len(labels)), accuracies, color=bar_colors)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_bar.set_ylabel("Best TM Accuracy (%)")
    ax_bar.set_title("Best TM Accuracy by Method")
    lo, hi = min(accuracies), max(accuracies)
    ax_bar.set_ylim(lo - (hi - lo) * 0.5, hi + (hi - lo) * 0.2)
    for bar, acc in zip(bars, accuracies):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fname = f"results_{dataset_name}_C{C}_T{T}_s{s}.png"
    out_path = os.path.join(save_path, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close()


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    """
    Findings:
    - at all clause levels, using the NN on top of the TM gives significant higher accuracy than the TM alone
    - low and high clause levels
    - alternating TM/NN is better than frozen TM at base
    - Good results at C=50, T=12.5, s=4.0, 15 rounds of 15. Also with C=20, T=5, s=3.0, 15 rounds of 15.
    - Gains are higher at low clause levels, marginal at high clause levels
    """

    C = 500
    T = C // 4
    s = 4.0
    number_of_state_bits = 8
    rounds = 5
    epochs_per_round = 25
    split_point = 0.3

    train = FashionMNIST(root="data", train=True, download=True)
    test = FashionMNIST(root="data", train=False, download=True)
    dataset_name = "FashionMNIST"
    train = EMNIST(root="data", train=True, download=True, split="letters")
    test = EMNIST(root="data", train=False, download=True, split="letters")
    dataset_name = "EMNIST-letters"
    train = MNIST(root="data", train=True, download=True)
    test = MNIST(root="data", train=False, download=True)
    dataset_name = "MNIST-1"

    x_train, y_train, x_test, y_test = binarize_dataset(train, test)
    # EMNIST letters labels are 1-indexed (1-26); remap to 0-indexed so the TM
    # allocates exactly 26 classes instead of 27

    per_epoch_df, summary_df = run_experiment(
        x_train, y_train, x_test, y_test,
        C=C, T=T, s=s,
        number_of_state_bits=number_of_state_bits,
        rounds=rounds,
        epochs_per_round=epochs_per_round,
        split_point=split_point,
        dataset_name=dataset_name,
        save_path="results",
    )