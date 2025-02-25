from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
from pickle import dump, load

from datasets import MNISTDataset, FashionMNISTDataset, KMNISTDataset, IMDBDataset,EMNISTLettersDataset

dataset = MNISTDataset()
X_train, Y_train, X_test, Y_test = dataset.get_data()

teacher_params = {
    "number_of_clauses": 1000,
    "T": 80,
    "s": 4.0,
    "boost_true_positive_feedback": 1,
    "number_of_state_bits": 8,
    "append_negated": True,
    "weighted_clauses": True
}

student_params = {
    "number_of_clauses": 100,
    "T": 10,
    "s": 4.0,
    "boost_true_positive_feedback": 1,
    "number_of_state_bits": 8,
    "append_negated": True,
    "weighted_clauses": True
}
teacher_epochs = 20
student_epochs = 40
skip_train_result = True
temperature = 2
print(f"Teacher params: {teacher_params}")
print(f"Student params: {student_params}")

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

print(f"Training baseline student ")
student = MultiClassTsetlinMachine(**student_params)
for i in range(student_epochs+teacher_epochs):
    test_result, train_result, train_time, train_result_time, test_time = normal_train_step(student, X_train, Y_train, X_test, Y_test, skip_train_result)
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

print(f"Training baseline teacher")
best_acc = 0
teacher = MultiClassTsetlinMachine(**teacher_params)
for i in range(teacher_epochs):#+student_epochs):
    test_result, train_result, train_time, train_result_time, test_time = normal_train_step(teacher, X_train, Y_train, X_test, Y_test, skip_train_result)
    if skip_train_result:
        print("#%02d Training: %.2fs || Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, test_time, test_result))
    else:
        print("#%02d Training: %.2fs || Accuracy (train) (%.2fs): %.2f%% | Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, train_result_time, train_result, test_time, test_result))

    if i <= teacher_epochs-1 and test_result > best_acc:
        best_acc = test_result
        print(f"Saving teacher checkpoint @ epoch {i+1} because of best accuracy {best_acc:.2f}%")
        dump(teacher, open("teacher_checkpoint.pkl", "wb"))

start_testing = time()
test_result = 100*(teacher.predict(X_test) == Y_test).mean()
stop_testing = time()
teacher_test_time = stop_testing-start_testing
baseline_teacher_acc = test_result
print(f"Baseline teacher accuracy: {baseline_teacher_acc:.2f}% ({teacher_test_time:.2f}s)")

print(f"Loading teacher checkpoint @ epoch {teacher_epochs}")

teacher = load(open("teacher_checkpoint.pkl", "rb"))

student = MultiClassTsetlinMachine(**student_params)
print(f"Initializing student with {student_params['number_of_clauses']} clauses from teacher")
student.init_from_teacher(teacher, student_params['number_of_clauses'], X_train, Y_train)
_, class_sums = teacher.predict_class_sums_2d(X_train)
print(f"Generating soft labels from teacher with temperature {temperature}")
soft_labels = teacher.get_soft_labels(X_train, temperature=temperature)


#import pdb; pdb.set_trace()

print("First 5 soft labels:")
for i in range(5):
    label_row = soft_labels[i]
    print(f" [" + " ".join([f"{x:.2e}" for x in label_row]) + "]")

print(f"Training student with soft labels")

for i in range(teacher_epochs, teacher_epochs+student_epochs):
    start_training = time()
    student.fit_soft(X_train, Y_train, soft_labels, epochs=1, incremental=True)
    stop_training = time()
    train_time = stop_training-start_training
    start_testing = time()
    test_result = 100*(student.predict(X_test) == Y_test).mean()
    stop_testing = time()
    test_result_time = stop_testing-start_testing

    # calculate accuracy
    if skip_train_result:
        print("#%02d Training: %.2fs || Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, test_time, test_result))
    else:
        start_train_result = time()
        train_result = 100*(student.predict(X_train) == Y_train).mean()
        stop_train_result = time()
        train_result_time = stop_train_result-start_train_result

        print("#%02d Training: %.2fs || Accuracy (train) (%.2fs): %.2f%% | Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, train_result_time, train_result, test_time, test_result))


start_testing = time()
test_result = 100*(student.predict(X_test) == Y_test).mean()
stop_testing = time()
distilled_test_time = stop_testing-start_testing
distilled_student_acc = test_result

print(f"Distilled student accuracy: {distilled_student_acc:.2f}% ({distilled_test_time:.2f}s)")

# now compare all accuracies and times
print("--------------------------------")
print(f"Student trained for {student_epochs+teacher_epochs} epochs")
print(f"Teacher trained for {student_epochs+teacher_epochs} epochs, checkpointed at epoch {teacher_epochs}")
print(f"Distilled student trained for {student_epochs} epochs with soft labels from teacher @ epoch {teacher_epochs}")
print(f"Student params: {student_params}")
print(f"Teacher params: {teacher_params}")
print(f"\nFinal scores:")
print(f" Baseline student accuracy: {baseline_student_acc:.2f}% ({student_test_time:.2f}s)")
print(f" Baseline teacher accuracy: {baseline_teacher_acc:.2f}% ({teacher_test_time:.2f}s)")
print(f" Distilled student accuracy: {distilled_student_acc:.2f}% ({distilled_test_time:.2f}s)")