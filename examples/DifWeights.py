import copy
import random
from time import perf_counter
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import os

@dataclass
class TMConfig:
    C: int
    T: int
    s: float
    weighted_clauses: bool
    number_of_state_bits: int

@dataclass
class TrainingResults:
    test_accuracy: list[float]
    test_time: list[float]
    train_time: list[float]

    def print_results(self):
        # print average times 
        print(f"test_acc={np.mean(self.test_accuracy):.2f}% | test_time={np.mean(self.test_time):.2f}s | train_time={np.mean(self.train_time):.2f}s / epochs={len(self.test_accuracy)}")

THRESHOLD = 75

def load_mnist_binary() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = MNIST(root="data", train=True, download=True)
    test = MNIST(root="data", train=False, download=True)

    x_train = np.where(train.data.numpy() > THRESHOLD, 1, 0).reshape(-1, 28 * 28).astype(np.uint32)
    y_train = train.targets.numpy().astype(np.uint32)
    x_test = np.where(test.data.numpy() > THRESHOLD, 1, 0).reshape(-1, 28 * 28).astype(np.uint32)
    y_test = test.targets.numpy().astype(np.uint32)
    return x_train, y_train, x_test, y_test

def train_tm(
    model: MultiClassTsetlinMachine,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
) -> Tuple[TrainingResults, MultiClassTsetlinMachine]:

    test_accuracy = []
    test_time = []
    train_time = []
    pbar = tqdm(range(epochs), desc=f"TM", dynamic_ncols=True, leave=False)
    for _ in pbar:
        train_start = perf_counter()
        model.fit(x_train, y_train, epochs=1, incremental=True)
        train_time.append(perf_counter() - train_start)
        test_start = perf_counter()
        test_accuracy.append(100.0 * (model.predict(x_test) == y_test).mean())
        test_time.append(perf_counter() - test_start)
        pbar.set_postfix(acc=f"{test_accuracy[-1]:.2f}%")
    pbar.close()
    return TrainingResults(test_accuracy, test_time, train_time)

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
    def __init__(self, n_classes, n_clauses, T: float = None, init_weights=None, learn_bias=False):
        super().__init__()

        self.n_classes = n_classes
        self.n_clauses = n_clauses
        self.T = T

        if init_weights is None:
            theta_init = torch.zeros(n_classes, n_clauses)
        else:
            init_weights = torch.as_tensor(init_weights, dtype=torch.float32)
            init_weights = torch.clamp(init_weights, min=1e-6)
            theta_init = torch.log(init_weights)

        self.theta = nn.Parameter(theta_init)

        if learn_bias:
            self.bias = nn.Parameter(torch.zeros(n_classes))
        else:
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
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    n_clauses: int,
    epochs: int,
    batch_size: int = 512,
    learning_rate: float = 1e-2,
    l2_weight: float = 1e-5,
    patience: int = 10,
) -> Tuple[TrainingResults, LogWeightHead]:
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    z_train_t = torch.as_tensor(z_train, dtype=torch.uint8)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    z_test_t = torch.as_tensor(z_test, dtype=torch.uint8)
    y_test_t = torch.as_tensor(y_test, dtype=torch.long)

    loader = DataLoader(TensorDataset(z_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(z_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_weight)

    test_accuracy = []
    test_time = []
    train_time = []
    best_acc = 0.0
    best_theta = model.theta.data.clone()
    epochs_no_improve = 0
    pbar = tqdm(range(epochs), desc=f"NN", dynamic_ncols=True, leave=False)
    for epoch in pbar:
        train_start = perf_counter()
        for zb, yb in loader:
            zb = zb.to(device)
            yb = yb.to(device)
            
            logits = model(zb)

            # compute loss
            loss = F.cross_entropy(logits, yb)
            weight_penalty = l2_weight * (model.weights ** 2).mean()

            loss = loss + weight_penalty

            # zero gradients
            optimizer.zero_grad()
            # backprop
            loss.backward()
            # step
            optimizer.step()
        
        train_time.append(perf_counter() - train_start)

        # now evaluate on test set
        with torch.no_grad():
            test_start = perf_counter()
            logits = model(z_test_t.to(device))
            acc = 100.0 * (logits.argmax(dim=1) == y_test_t.to(device)).float().mean().item()
            test_accuracy.append(acc)
            test_time.append(perf_counter() - test_start)
            if acc > best_acc:
                best_acc = acc
                best_theta = model.theta.data.clone()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            tqdm.write(f"Test accuracy: {acc:.2f}%, loss: {loss.item():.2f}")
            pbar.set_postfix(acc=f"{acc:.2f}%")

        if epochs_no_improve >= patience:
            tqdm.write(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            break

    model.theta.data.copy_(best_theta)
    return TrainingResults(test_accuracy, test_time, train_time)

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    C = 800
    T = 200
    s = 4.0
    number_of_state_bits = 8
    weighted_clauses = False

    # get MNIST data
    x_train, y_train, x_test, y_test = load_mnist_binary()

    # Train unweighted TM and save to pickle
    unweighted_tm_path = f"unweighted_tm_C{C}_T{T}_s{s}.pkl"
    if os.path.exists(unweighted_tm_path):
        unweighted_tm = pkl.load(open(unweighted_tm_path, "rb"))
    else:
        print(f"Training unweighted TM and saving to {unweighted_tm_path}")
        unweighted_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=False)
        unweighted_results = train_tm(unweighted_tm, x_train, y_train, x_test, y_test, epochs=50)
        unweighted_results.print_results() # around 88% for C=300, T=100, s=4.0, weighted_clauses=False
        pkl.dump(unweighted_tm, open(unweighted_tm_path, "wb"))

    ## Train weighted TM and save to pickle
    weighted_tm_path = f"weighted_tm_C{C}_T{T}_s{s}.pkl"
    if os.path.exists(weighted_tm_path):
        weighted_tm = pkl.load(open(weighted_tm_path, "rb"))
    else:
        print(f"Training weighted TM and saving to {weighted_tm_path}")
        weighted_tm = MultiClassTsetlinMachine(C, T, s, number_of_state_bits=number_of_state_bits, weighted_clauses=True)
        weighted_results = train_tm(weighted_tm, x_train, y_train, x_test, y_test, epochs=50)
        weighted_results.print_results() # around 96% for C=300, T=100, s=4.0, weighted_clauses=True
        pkl.dump(weighted_tm, open(weighted_tm_path, "wb"))
    

    # train neural network with frozen TM
    # column k = i * number_of_clauses + j is the feature for class i, clause j.
    Z_train = weighted_tm.transform(x_train, inverted=False).astype("uint8")
    Z_test = weighted_tm.transform(x_test, inverted=False).astype("uint8")
    print(f"Unweighted TM test accuracy: {100.0 * (unweighted_tm.predict(x_test) == y_test).mean():.2f}%")
    print(f"Weighted TM test accuracy: {100.0 * (weighted_tm.predict(x_test) == y_test).mean():.2f}%")
    
    # init weights from weighted TM
    init_weights = weighted_tm.get_clause_weights()
    # print range, mean of weights in the same line
    print(f"Init weights range: {init_weights.min():.2f} to {init_weights.max():.2f}, mean: {init_weights.mean():.2f}")

    nn_model = LogWeightHead(n_classes=10, n_clauses=weighted_tm.number_of_clauses, init_weights=init_weights)
    nn_results = train_log_weight_head(nn_model, Z_train, y_train, Z_test, y_test, n_classes=10, n_clauses=weighted_tm.number_of_clauses, epochs=50)
    nn_results.print_results()

    # scale weights so TM clamp never activates, then write back
    scaled_weights = scale_weights_for_tm(nn_model, Z_test, weighted_tm.T)
    weighted_tm.set_clause_weights(scaled_weights)
    print(f"Weighted TM test accuracy: {100.0 * (weighted_tm.predict(x_test) == y_test).mean():.2f}%")
    
    
    # now do the same for unweighted TM
    Z_train = unweighted_tm.transform(x_train, inverted=False).astype("uint8")
    Z_test = unweighted_tm.transform(x_test, inverted=False).astype("uint8")
    
    nn_model = LogWeightHead(n_classes=10, n_clauses=unweighted_tm.number_of_clauses, T=unweighted_tm.T)

    nn_results = train_log_weight_head(nn_model, Z_train, y_train, Z_test, y_test, n_classes=10, n_clauses=unweighted_tm.number_of_clauses, epochs=150)
    nn_results.print_results()
    scaled_weights = scale_weights_for_tm(nn_model, Z_test, unweighted_tm.T)
    unweighted_tm.set_clause_weights(scaled_weights)
    print(f"Unweighted TM (with NN weights) test accuracy: {100.0 * (unweighted_tm.predict(x_test) == y_test).mean():.2f}%")