from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
from pickle import dump, load

from keras.datasets import mnist, fashion_mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

teacher_params = (400, 10, 10)
student_params = (100, 10, 10)

teacher = MultiClassTsetlinMachine(*teacher_params)
teacher_epochs = 20
student_epochs = 30

print(f"Training baseline student")
student = MultiClassTsetlinMachine(*student_params)
for i in range(student_epochs+teacher_epochs):
    start_training = time()
    student.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result = 100*(student.predict(X_test) == Y_test).mean()
    stop_testing = time()

    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

baseline_student_acc = result

print(f"Training baseline teacher")
teacher = MultiClassTsetlinMachine(*teacher_params)
for i in range(student_epochs+teacher_epochs):
    start_training = time()
    teacher.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result = 100*(teacher.predict(X_test) == Y_test).mean()
    stop_testing = time()

    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
    if i == teacher_epochs-1:
        print("Saving teacher checkpoint")
        dump(teacher, open("teacher_checkpoint.pkl", "wb"))

baseline_teacher_acc = result

teacher = load(open("teacher_checkpoint.pkl", "rb"))
soft_labels = teacher.get_soft_labels(X_train, temperature=2.0)

print(f"Training student with soft labels")
student = MultiClassTsetlinMachine(*student_params)
for i in range(student_epochs):
    start_training = time()
    student.fit_soft(X_train, Y_train, soft_labels, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    result = 100*(student.predict(X_test) == Y_test).mean()
    stop_testing = time()

    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
