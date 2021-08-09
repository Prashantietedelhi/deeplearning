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

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes  = load_data()

## PReprocessing
m = train_set_x_orig.shape[0]
n  = train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3]

train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x = train_set_x/255
test_set_x = test_set_x/255


X = train_set_x
Y = train_set_y_orig
w = np.zeros((n,1))
b= 0
dw = w
db = b
learning_rate = 0.005

def sigmoid(z):
    return 1/(1+np.exp(-z))

for i in range(2000):
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -np.sum(((Y*np.log(A)) + ((1-Y)*(np.log(1-A)))))/m
    # -np.sum(((Y * np.log(A)) + ((1 - Y) * (np.log(1 - A))))) / m
    dw = np.dot(X, (A-Y).T)/m
    db =     np.sum(A-Y)/m
    w =  w - (learning_rate*dw)
    b= b - (learning_rate*db)

    if   i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
print(w)
print(b)

def predict(w, b, data):
    predictvalue =np.zeros((1,data.shape[1]))
    A = sigmoid(np.dot(w.T, data)+b)
    for i in range((A.shape[1])):
        if A[:,i]>0.5:
            predictvalue[:,i]  = 1

    return predictvalue

Y_prediction_train = predict(w,b,train_set_x)
Y_prediction_test  = predict(w,b,test_set_x)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y_orig)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y_orig)) * 100))


