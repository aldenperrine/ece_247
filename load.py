import numpy as np

"""
data shapes:

X_test (443, 22, 1000)
y_test (443,)
person_train_valid (2115, 1)
X_train_valid (2115, 22, 1000)
y_train_valid (2115,)
person_test (443, 1)
"""


def load_data():
    X_test = np.load("project/X_test.npy")
    y_test = np.load("project/y_test.npy")
    person_train_valid = np.load("project/person_train_valid.npy")
    X_train_valid = np.load("project/X_train_valid.npy")
    y_train_valid = np.load("project/y_train_valid.npy")
    person_test = np.load("project/person_test.npy")
    return (X_test, y_test, person_train_valid, X_train_valid, y_train_valid, person_test)


if __name__ == "__main__":
    sets = load_data()
    labels = ["X_test",
              "y_test",
              "person_train_valid",
              "X_train_valid",
              "y_train_valid",
              "person_test"]
    for i, k in enumerate(sets):
        print('{}: {} '.format(labels[i], k.shape))
