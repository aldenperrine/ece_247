import numpy as np

def load_data():
    X_test = np.load("project/X_test.npy")
    y_test = np.load("project/y_test.npy")
    person_train_valid = np.load("project/person_train_valid.npy")
    X_train_valid = np.load("project/X_train_valid.npy")
    y_train_valid = np.load("project/y_train_valid.npy")
    person_test = np.load("project/person_test.npy")
    return (X_test, y_test, person_train_valid, X_train_valid, y_train_valid, person_test)

def load_data_subject_1_train_and_test():
    X_test = np.load("project/X_test.npy")
    y_test = np.load("project/y_test.npy")
    person_train_valid = np.load("project/person_train_valid.npy")
    X_train_valid = np.load("project/X_train_valid.npy")
    y_train_valid = np.load("project/y_train_valid.npy")
    person_test = np.load("project/person_test.npy")

    subject_1_count = 0
    for i in range(person_train_valid.shape[0]):
        if person_train_valid[i] == 0:
            subject_1_count += 1
    subject_1_valid = [0] * subject_1_count
    j = 0
    for i in range(person_train_valid.shape[0]):
        if person_train_valid[i] == 0:
            subject_1_valid[j] = i
            j += 1
    # print(f'There are {subject_1_count} trials for subject 1')
    # print(f'Subject 1 is involved in trials {[x for x in subject_1_valid]}')
    s1_x_valid = np.zeros((subject_1_count, 22, 1000))
    r = 0
    for i in subject_1_valid:
        s1_x_valid[r,:,:] = X_train_valid[i,:,:]
        r += 1
    # print(f's1_x_valid shape is {s1_x_valid.shape}')
    s1_y_valid = np.zeros((subject_1_count))
    r = 0
    for p in subject_1_valid:
        s1_y_valid[r] = y_train_valid[p]
        r += 1
    # print(f's1_y_valid shape is {s1_y_valid.shape}')

    # print('')
    # print('test data below')

    subject_1_count_test = 0
    for i in range(person_test.shape[0]):
        if person_test[i] == 0:
            subject_1_count_test += 1
    subject_1_valid_test = [0] * subject_1_count_test
    j = 0
    for i in range(person_test.shape[0]):
        if person_test[i] == 0:
            subject_1_valid_test[j] = i
            j += 1
    # print(f'There are {subject_1_count_test} trials for subject 1')
    # print(f'Subject 1 is involved in trials {[x for x in subject_1_valid_test]}')
    s1_x_test = np.zeros((subject_1_count_test, 22, 1000))
    r = 0
    for i in subject_1_valid_test:
        s1_x_test[r,:,:] = X_test[i,:,:]
        r += 1
    # print(f's1_x_test shape is {s1_x_test.shape}')
    s1_y_test = np.zeros((subject_1_count_test))
    r = 0
    for p in subject_1_valid_test:
        s1_y_test[r] = y_test[p]
        r += 1
    # print(f's1_y_test shape is {s1_y_test.shape}')
    return (s1_x_test, s1_y_test, person_train_valid, s1_x_valid, s1_y_valid, person_test)

def load_data_subject_1_test_and_full_train():
    X_test = np.load("project/X_test.npy")
    y_test = np.load("project/y_test.npy")
    person_train_valid = np.load("project/person_train_valid.npy")
    X_train_valid = np.load("project/X_train_valid.npy")
    y_train_valid = np.load("project/y_train_valid.npy")
    person_test = np.load("project/person_test.npy")

    subject_1_count_test = 0
    for i in range(person_test.shape[0]):
        if person_test[i] == 0:
            subject_1_count_test += 1
    subject_1_valid_test = [0] * subject_1_count_test
    j = 0
    for i in range(person_test.shape[0]):
        if person_test[i] == 0:
            subject_1_valid_test[j] = i
            j += 1
    # print(f'There are {subject_1_count_test} trials for subject 1')
    # print(f'Subject 1 is involved in trials {[x for x in subject_1_valid_test]}')
    s1_x_test = np.zeros((subject_1_count_test, 22, 1000))
    r = 0
    for i in subject_1_valid_test:
        s1_x_test[r,:,:] = X_test[i,:,:]
        r += 1
    # print(f's1_x_test shape is {s1_x_test.shape}')
    s1_y_test = np.zeros((subject_1_count_test))
    r = 0
    for p in subject_1_valid_test:
        s1_y_test[r] = y_test[p]
        r += 1
    # print(f's1_y_test shape is {s1_y_test.shape}')
    return (s1_x_test, s1_y_test, person_train_valid, X_train_valid, y_train_valid, person_test)


if __name__ == "__main__":
    sets = load_data()
    labels = ["X_test",
    "y_test",
    "person_train_valid",
    "X_train_valid",
    "y_train_valid",
    "person_test"]
    print('generating data using all test and all train')
    for i, k in enumerate(sets):
        print('{}: {} '.format(labels[i], k.shape))
    print('generating data using subj 1 test and subj 1 train')
    printy = load_data_subject_1_train_and_test()
    for i, k in enumerate(printy):
        print('{}: {} '.format(labels[i], k.shape))
    print('generating data using subj 1 test and full train')
    thingy = load_data_subject_1_test_and_full_train()
    for i, k in enumerate(thingy):
        print('{}: {} '.format(labels[i], k.shape))