from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 

tm = MultiClassTsetlinMachine(2000, 50, 10.0)
epochs = 50

print(f"\nAccuracy over {epochs} epochs:\n")
for i in range(epochs):
    start_training = time()
    tm.fit(X_train, Y_train, epochs=1, incremental=True)
    stop_training = time()

    start_testing = time()
    prediction, class_sums = tm.predict_class_sums_2d(X_test, return_class_sums=True)
    result = 100*(prediction == Y_test).mean()
    stop_testing = time()
    
    print(f"Class sums:\n{class_sums}")

    print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))