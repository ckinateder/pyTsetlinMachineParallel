import numpy as np
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine

# Create synthetic data
np.random.seed(42)
X = np.random.randint(0, 2, size=(1000, 200))  # 1000 samples, 200 binary features
y = np.zeros(1000, dtype=np.int32)

# Create some simple rules for class assignment
y[X[:, 0] & X[:, 1]] = 1  # Class 1 if first two features are 1
y[X[:, 2] & X[:, 3]] = 2  # Class 2 if features 2 and 3 are 1

# Create soft labels with some noise
soft_labels = np.zeros((1000, 3))  # 3 classes
for i in range(1000):
    true_class = y[i]
    # Add some uncertainty to the labels
    probs = np.random.dirichlet([1, 1, 1])  # Random probabilities that sum to 1
    # Make true class more likely
    probs = 0.8 * np.eye(3)[true_class] + 0.2 * probs
    soft_labels[i] = probs

# Split into train and test
train_idx = np.random.choice(1000, 800, replace=False)
test_idx = np.array(list(set(range(1000)) - set(train_idx)))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
soft_labels_train = soft_labels[train_idx]

# Create and train TM with hard labels
tm_hard = MultiClassTsetlinMachine(
    number_of_clauses=100,
    T=15,
    s=3.9,
    boost_true_positive_feedback=1,
    number_of_state_bits=8
)
tm_hard.fit(X_train, y_train, epochs=50)
hard_acc = np.mean(tm_hard.predict(X_test) == y_test)

# Create and train TM with soft labels
tm_soft = MultiClassTsetlinMachine(
    number_of_clauses=100,
    T=15,
    s=3.9,
    boost_true_positive_feedback=1,
    number_of_state_bits=8
)
tm_soft.fit_soft(X_train, y_train, soft_labels_train, epochs=50)
soft_acc = np.mean(tm_soft.predict(X_test) == y_test)

print(f"Hard labels accuracy: {hard_acc:.3f}")
print(f"Soft labels accuracy: {soft_acc:.3f}")

# Test with different noise levels
for noise in [0.1, 0.3, 0.5, 0.7]:
    soft_labels_noisy = np.zeros_like(soft_labels)
    for i in range(len(soft_labels)):
        true_class = y[i]
        probs = np.random.dirichlet([1, 1, 1])
        # Mix true label with noise
        soft_labels_noisy[i] = (1 - noise) * np.eye(3)[true_class] + noise * probs
    
    soft_labels_train_noisy = soft_labels_noisy[train_idx]
    
    tm_soft_noisy = MultiClassTsetlinMachine(
        number_of_clauses=100,
        T=15,
        s=3.9,
        boost_true_positive_feedback=1,
        number_of_state_bits=8
    )
    tm_soft_noisy.fit_soft(X_train, y_train, soft_labels_train_noisy, epochs=50)
    noisy_acc = np.mean(tm_soft_noisy.predict(X_test) == y_test)
    
    print(f"Soft labels accuracy (noise={noise:.1f}): {noisy_acc:.3f}")



from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
from pickle import dump, load

from keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = np.where(X_train.reshape((X_train.shape[0], 28*28)) > 75, 1, 0) 
X_test = np.where(X_test.reshape((X_test.shape[0], 28*28)) > 75, 1, 0) 
ts = (20, 10)
student_num_clauses = 100
teacher_num_clauses = 800
teacher = MultiClassTsetlinMachine(teacher_num_clauses, *ts)
student = MultiClassTsetlinMachine(student_num_clauses, *ts)
epochs = 10

#"""
print(f"Training teacher")
for i in range(epochs):
	start_training = time()
	teacher.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(teacher.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

dump(teacher, open("teacher.pkl", "wb"))
#"""
teacher = load(open("teacher.pkl", "rb"))
soft_labels = teacher.get_soft_labels(X_train, temperature=2.0)

print(f"Training student with soft labels")
for i in range(epochs):
	start_training = time()
	student.fit_soft(X_train, Y_train, soft_labels, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(student.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

print(f"Training baseline student")
student = MultiClassTsetlinMachine(student_num_clauses, *ts)
for i in range(epochs):
	start_training = time()
	student.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(student.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
