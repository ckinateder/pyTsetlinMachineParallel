import copy
import random
from time import perf_counter

import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from torchvision.datasets import MNIST
from tqdm import tqdm


TEACHER_PARAMS = {"C": 1000, "T": 10, "s": 4.0, "epochs": 20}
STUDENT_PARAMS = {"C": 100, "T": 10, "s": 4.0, "epochs": 40}
TEMPERATURE = 3.0
ALPHA = 0.5
Z = 0.3
THRESHOLD = 75
WEIGHTED_CLAUSES = True
NUMBER_OF_STATE_BITS = 8


def load_mnist_binary():
    train = MNIST(root="data", train=True, download=True)
    test = MNIST(root="data", train=False, download=True)

    x_train = np.where(train.data.numpy() > THRESHOLD, 1, 0).reshape(-1, 28 * 28).astype(np.uint32)
    y_train = train.targets.numpy().astype(np.uint32)
    x_test = np.where(test.data.numpy() > THRESHOLD, 1, 0).reshape(-1, 28 * 28).astype(np.uint32)
    y_test = test.targets.numpy().astype(np.uint32)
    return x_train, y_train, x_test, y_test


def run_epoch(model, x_train, y_train, x_test, y_test, *, soft_labels=None):
    train_start = perf_counter()
    if soft_labels is None:
        model.fit(x_train, y_train, epochs=1, incremental=True)
    else:
        model.fit_soft(
            x_train,
            y_train,
            epochs=1,
            incremental=True,
            soft_labels=soft_labels,
            temperature=TEMPERATURE,
            alpha=ALPHA,
        )
    train_time = perf_counter() - train_start

    test_start = perf_counter()
    test_acc = 100.0 * (model.predict(x_test) == y_test).mean()
    test_time = perf_counter() - test_start
    return test_acc, train_time, test_time


def train_model(name, model, epochs, x_train, y_train, x_test, y_test, *, soft_labels=None):
    pbar = tqdm(range(epochs), desc=name, dynamic_ncols=True)
    acc_history = []
    for _ in pbar:
        test_acc, train_time, test_time = run_epoch(
            model, x_train, y_train, x_test, y_test, soft_labels=soft_labels
        )
        acc_history.append(test_acc)
        pbar.set_postfix(acc=f"{test_acc:.2f}%")
    pbar.close()
    return acc_history


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    x_train, y_train, x_test, y_test = load_mnist_binary()
    print(f"Loaded MNIST: train={x_train.shape}, test={x_test.shape}, threshold={THRESHOLD}/255")

    teacher = MultiClassTsetlinMachine(
        TEACHER_PARAMS["C"],
        TEACHER_PARAMS["T"],
        TEACHER_PARAMS["s"],
        number_of_state_bits=NUMBER_OF_STATE_BITS,
        weighted_clauses=WEIGHTED_CLAUSES,
    )
    baseline = MultiClassTsetlinMachine(
        STUDENT_PARAMS["C"],
        STUDENT_PARAMS["T"],
        STUDENT_PARAMS["s"],
        number_of_state_bits=NUMBER_OF_STATE_BITS,
        weighted_clauses=WEIGHTED_CLAUSES,
    )
    student = MultiClassTsetlinMachine(
        STUDENT_PARAMS["C"],
        STUDENT_PARAMS["T"],
        STUDENT_PARAMS["s"],
        number_of_state_bits=NUMBER_OF_STATE_BITS,
        weighted_clauses=WEIGHTED_CLAUSES,
    )

    print("\nTraining teacher...")
    best_teacher = None
    best_teacher_acc = -1.0
    teacher_acc = []
    teacher_pbar = tqdm(range(TEACHER_PARAMS["epochs"]), desc="Teacher", dynamic_ncols=True)
    for _ in teacher_pbar:
        test_acc, train_time, test_time = run_epoch(teacher, x_train, y_train, x_test, y_test)
        teacher_acc.append(test_acc)
        teacher_pbar.set_postfix(acc=f"{test_acc:.2f}%")
        if test_acc > best_teacher_acc:
            best_teacher_acc = test_acc
            best_teacher = copy.deepcopy(teacher)
    teacher_pbar.close()

    print("\nTraining baseline (same student size, no KD)...")
    baseline_acc = train_model(
        "Baseline", baseline, TEACHER_PARAMS["epochs"] + STUDENT_PARAMS["epochs"], x_train, y_train, x_test, y_test
    )

    print("\nInitializing student from teacher and generating soft labels...")
    student.init_from_teacher(best_teacher, x_train, y_train, clauses_per_class=STUDENT_PARAMS["C"], z=Z)
    soft_labels = best_teacher.get_soft_labels(x_train)

    print("\nTraining student with KD...")
    student_acc = train_model(
        "Student", student, STUDENT_PARAMS["epochs"], x_train, y_train, x_test, y_test, soft_labels=soft_labels
    )

    print("\nFinal test accuracy:")
    print(f"Teacher (best over {TEACHER_PARAMS['epochs']} epochs): {best_teacher_acc:.2f}%")
    print(f"Baseline (last epoch): {baseline_acc[-1]:.2f}%")
    print(f"Student KD (last epoch): {student_acc[-1]:.2f}%")
