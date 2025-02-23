from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
from pickle import dump, load

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

teacher_params = (200, 10, 10)
student_params = (50, 10, 10)

teacher = MultiClassTsetlinMachine(*teacher_params)
teacher_epochs = 10
student_epochs = 10
skip_train_result = True
temperature = 4
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

print(f"Training baseline student for {student_epochs} epochs")
student = MultiClassTsetlinMachine(*student_params)
for i in range(student_epochs+teacher_epochs):
    test_result, train_result, train_time, train_result_time, test_time = normal_train_step(student, X_train, Y_train, X_test, Y_test, skip_train_result)
    if skip_train_result:
        print("#%02d Training: %.2fs || Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, test_time, test_result))
    else:
        print("#%02d Training: %.2fs || Accuracy (train) (%.2fs): %.2f%% | Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, train_result_time, train_result, test_time, test_result))

baseline_student_acc = test_result

print(f"Training baseline teacher for {teacher_epochs} epochs")
teacher = MultiClassTsetlinMachine(*teacher_params)
for i in range(teacher_epochs+student_epochs):
    test_result, train_result, train_time, train_result_time, test_time = normal_train_step(teacher, X_train, Y_train, X_test, Y_test, skip_train_result)
    if skip_train_result:
        print("#%02d Training: %.2fs || Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, test_time, test_result))
    else:
        print("#%02d Training: %.2fs || Accuracy (train) (%.2fs): %.2f%% | Accuracy (test) (%.2fs): %.2f%%" % (i+1, train_time, train_result_time, train_result, test_time, test_result))

    if i == teacher_epochs-1:
        print("Saving teacher checkpoint")
        dump(teacher, open("teacher_checkpoint.pkl", "wb"))

baseline_teacher_acc = test_result

print(f"Loading teacher checkpoint @ epoch {teacher_epochs}")

teacher = load(open("teacher_checkpoint.pkl", "rb"))

student = MultiClassTsetlinMachine(*student_params)
print(f"Initializing student with {student_params[0]} clauses from teacher")
student.init_from_teacher(teacher, student_params[0], X_train, Y_train)
print(f"Generating soft labels from teacher with temperature {temperature}")
soft_labels = teacher.get_soft_labels(X_train, temperature=temperature)
print(f"Training student with soft labels")

for i in range(student_epochs, teacher_epochs+student_epochs):
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
