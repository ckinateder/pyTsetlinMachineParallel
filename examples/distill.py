from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
from pickle import dump, load
import os
import json
from datasets import MNISTDataset, FashionMNISTDataset, KMNISTDataset, IMDBDataset, EMNISTLettersDataset

# Configuration
dataset = KMNISTDataset()
X_train, Y_train, X_test, Y_test = dataset.get_data()

# take a subset of the data
trim = 1 # 0.5 = 50% of the data
X_train = X_train[:int(len(X_train)*trim)]
Y_train = Y_train[:int(len(Y_train)*trim)]
X_test = X_test[:int(len(X_test)*trim)]
Y_test = Y_test[:int(len(Y_test)*trim)]

teacher_params = {
    "number_of_clauses": 1000,
    "T": 100,
    "s": 8.2,
    "boost_true_positive_feedback": 1,
    "number_of_state_bits": 8,
    "append_negated": True,
    "weighted_clauses": True
}

student_params = {
    "number_of_clauses": 100,
    "T": 100,
    "s": 8.2,
    "boost_true_positive_feedback": 1,
    "number_of_state_bits": 8,
    "append_negated": True,
    "weighted_clauses": True
}

# Training hyperparameters
teacher_epochs = 50
student_epochs = 100
skip_train_result = True

# Distillation hyperparameters
temperature = 4.0  # Higher temperature for better distillation
alpha = 0.5  # Controls balance between hard and soft labels (0=all soft, 1=all hard)

print(f"Teacher params: {teacher_params}")
print(f"Student params: {student_params}")
print(f"Distillation params: temperature={temperature}, alpha={alpha}")

def normal_train_step(model, X_train, Y_train, X_test, Y_test, skip_train_result=False):        
    start_training = time()
    model.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    test_result = 100*(model.predict(X_test) == Y_test).mean()
    stop_testing = time()

    if skip_train_result:
        return test_result, 0, stop_training-start_training, 0, stop_testing-start_testing
    
    else:
        start_train_result = time()
        train_result = 100*(model.predict(X_train) == Y_train).mean()
        stop_train_result = time()

    return test_result, train_result, stop_training-start_training, stop_train_result-start_train_result, stop_testing-start_testing

# Create results directory
os.makedirs("results", exist_ok=True)

# Train baseline student
print(f"Training baseline student ")
student = MultiClassTsetlinMachine(**student_params)
student_history = []

for i in range(student_epochs+teacher_epochs):
    test_result, train_result, train_time, train_result_time, test_time = normal_train_step(student, X_train, Y_train, X_test, Y_test, skip_train_result)
    student_history.append(test_result)
    
    if skip_train_result:
        print("#%02d Training: %.2fs || Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, test_time, test_result))
    else:
        print("#%02d Training: %.2fs || Accuracy (train) (%.2fs): %.2f%% | Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, train_result_time, train_result, test_time, test_result))

start_testing = time()
test_result = 100*(student.predict(X_test) == Y_test).mean()
stop_testing = time()
student_test_time = stop_testing-start_testing
print(f"Baseline student accuracy: {test_result:.2f}% ({student_test_time:.2f}s)")
baseline_student_acc = test_result

# Train the teacher model
print(f"Training baseline teacher")
best_acc = 0
teacher = MultiClassTsetlinMachine(**teacher_params)
teacher_history = []

for i in range(teacher_epochs):
    test_result, train_result, train_time, train_result_time, test_time = normal_train_step(teacher, X_train, Y_train, X_test, Y_test, skip_train_result)
    teacher_history.append(test_result)
    
    if skip_train_result:
        print("#%02d Training: %.2fs || Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, test_time, test_result))
    else:
        print("#%02d Training: %.2fs || Accuracy (train) (%.2fs): %.2f%% | Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, train_result_time, train_result, test_time, test_result))

    if i <= teacher_epochs-1 and test_result > best_acc:
        best_acc = test_result
        print(f"Saving teacher checkpoint @ epoch {i+1} because of best accuracy {best_acc:.2f}%")
        dump(teacher, open("results/teacher_checkpoint.pkl", "wb"))

start_testing = time()
test_result = 100*(teacher.predict(X_test) == Y_test).mean()
stop_testing = time()
teacher_test_time = stop_testing-start_testing
baseline_teacher_acc = test_result
print(f"Baseline teacher accuracy: {baseline_teacher_acc:.2f}% ({teacher_test_time:.2f}s)")

# Load best teacher checkpoint
print(f"Loading teacher checkpoint")
teacher = load(open("results/teacher_checkpoint.pkl", "rb"))

# Initialize student from teacher
distilled = MultiClassTsetlinMachine(**student_params)
print(f"Initializing student with {student_params['number_of_clauses']} clauses from teacher")
distilled.init_from_teacher(teacher, student_params['number_of_clauses'], X_train, Y_train)

# Generate soft labels with temperature
print(f"Generating soft labels from teacher with temperature {temperature}")
soft_labels = teacher.get_soft_labels(X_train)#, temperature=temperature)

# Print first few soft labels to check distribution
print("First 5 soft labels:")
for i in range(5):
    label_row = soft_labels[i]
    print(f" [" + " ".join([f"{x:.2e}" for x in label_row]) + "]")
print("First 5 output probabilities:")

# Train distilled model with soft la    bels
print(f"Training student with soft labels (alpha={alpha}, temperature={temperature})")
distilled_history = []

for i in range(teacher_epochs, teacher_epochs+student_epochs):
    start_training = time()
    distilled.fit_soft(X_train, Y_train, soft_labels, epochs=1, incremental=True, alpha=alpha, temperature=temperature)
    stop_training = time()
    train_time = stop_training-start_training
    
    start_testing = time()
    test_result = 100*(distilled.predict(X_test) == Y_test).mean()
    stop_testing = time()
    test_time = stop_testing-start_testing
    distilled_history.append(test_result)

    # calculate accuracy
    if skip_train_result:
        print("#%02d Training: %.2fs || Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, test_time, test_result))
    else:
        start_train_result = time()
        train_result = 100*(distilled.predict(X_train) == Y_train).mean()
        stop_train_result = time()
        train_result_time = stop_train_result-start_train_result

        print("#%02d Training: %.2fs || Accuracy (train) (%.2fs): %.2f%% | Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, train_result_time, train_result, test_time, test_result))

# Final evaluation
start_testing = time()
test_result = 100*(distilled.predict(X_test) == Y_test).mean()
stop_testing = time()
distilled_test_time = stop_testing-start_testing
distilled_student_acc = test_result

print(f"Distilled student accuracy: {distilled_student_acc:.2f}% ({distilled_test_time:.2f}s)")

# Save results
results = {
    "teacher_history": teacher_history,
    "student_history": student_history,
    "distilled_history": distilled_history,
    "avg_teacher_acc": np.mean(teacher_history),
    "avg_student_acc": np.mean(student_history),
    "avg_distilled_acc": np.mean(distilled_history),
    "teacher_params": teacher_params,
    "student_params": student_params,
    "distillation_params": {
        "temperature": temperature,
        "alpha": alpha
    },
    "final_accuracies": {
        "teacher": baseline_teacher_acc,
        "student": baseline_student_acc,
        "distilled": distilled_student_acc
    },
    "inference_times": {
        "teacher": teacher_test_time,
        "student": student_test_time,
        "distilled": distilled_test_time
    }
}
# save results as json
with open("results/distillation_results.json", "w") as f:
    json.dump(results, f)

# Generate activation maps
try:
    from ActivationMaps import visualize_activation_maps
    print("Generating activation maps...")
    samples = np.random.randint(0, len(X_test), size=4)
    visualize_activation_maps(teacher, student, distilled, X_test[samples], (28, 28), "results/activation_map.png")
except Exception as e:
    raise e
    print(f"Could not generate activation maps: {e}")

# Print summary
print("--------------------------------")
print(f"Student trained for {student_epochs+teacher_epochs} epochs")
print(f"Teacher trained for {teacher_epochs} epochs")
print(f"Distilled student trained for {student_epochs} epochs (alpha={alpha}, temperature={temperature})")
print(f"Student params: {student_params}")
print(f"Teacher params: {teacher_params}")
print(f"\nFinal scores:")
print(f" Baseline student accuracy: {baseline_student_acc:.2f}% ({student_test_time:.2f}s)")
print(f" Baseline teacher accuracy: {baseline_teacher_acc:.2f}% ({teacher_test_time:.2f}s)")
print(f" Distilled student accuracy: {distilled_student_acc:.2f}% ({distilled_test_time:.2f}s)")
print(f"\nSpeedup: {teacher_test_time/distilled_test_time:.2f}x")
print(f"Average teacher accuracy: {results['avg_teacher_acc']:.2f}%")
print(f"Average student accuracy: {results['avg_student_acc']:.2f}%")
print(f"Average distilled accuracy: {results['avg_distilled_acc']:.2f}%")
