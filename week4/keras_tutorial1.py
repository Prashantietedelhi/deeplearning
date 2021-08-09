import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    # train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    # test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes  = load_data()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1)

train_set_x_flatten = train_set_x_flatten/255
test_set_x_flatten = test_set_x_flatten/255
n_features = train_set_x_flatten.shape[1]
print(n_features)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(20, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(7, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(5, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_set_x_flatten, train_set_y_orig, epochs=150, batch_size=32, verbose=0)

loss, acc = model.evaluate(test_set_x_flatten, test_set_y_orig, verbose=0)
print(loss,acc)