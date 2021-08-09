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
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x_flatten = train_set_x_flatten/255
test_set_x_flatten = test_set_x_flatten/255
layers_dim = [12288, 20, 7, 5, 1]
lr  =  0.0075
### initialize default parameters
def initialize_parameters_deep(layers_dim):
    np.random.seed(3)
    L = len(layers_dim)
    parameters = {}
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers_dim[l],layers_dim[l-1])*np.sqrt(2/layers_dim[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dim[l],1))
    return parameters

parameters = (initialize_parameters_deep(layers_dim))

#### Shape of parameters
# for k,v in parameters.items():
#     print(k, v.shape)

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0.0, Z)

def cost(Y, AL):
    m = Y.shape[1]
    c =  -(np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL))))/m
    c = np.squeeze(c)
    return c
def sigmoid_backward(dA, Z):
    s = sigmoid(Z)
    dZ = dA*s*(1-s)
    return dZ
def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

A = train_set_x_flatten
Y =train_set_y_orig
L = len(layers_dim)
m = Y.shape[1]
parameters["A" + str(0)] = train_set_x_flatten
# np.random.seed(1)
for i in range(1,20):
    ## Forward Propagation
    for l in range(1,L-1):
        Z = np.dot(parameters["W"+str(l)], parameters["A" + str(l-1)])+(parameters["b"+str(l)])
        parameters["Z"+str(l)] = Z
        parameters["A" + str(l)] = relu(Z)

    Z = np.dot(parameters["W"+str(L-1)], parameters["A" + str(L-2)])+(parameters["b"+str(L-1)])
    parameters["A" + str(L - 1)] = sigmoid(Z)
    parameters["Z" + str(L-1)] = Z


    ## calculate Cost
    costval = cost(Y, parameters["A" + str(L - 1)])
    print(costval)
    ## Backward Propagation
    # dAL = - (np.divide(Y, parameters["A" + str(L - 1)]) - np.divide(1 - Y, 1 - parameters["A" + str(L - 1)]))
    for l in reversed(range(1,L)):
        if l == L - 1:
            dAL =  - (np.divide(Y, parameters["A" + str(L - 1)]) - np.divide(1 - Y, 1 - parameters["A" + str(L - 1)]))
            dZ = sigmoid_backward(dAL, parameters["Z" + str(L - 1)])
            dw = np.dot(dZ, parameters["A" + str(l - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dAL = np.dot(parameters["W" + str(l)].T, dZ)
        else:
            dZ = relu_backward(dAL, parameters["Z" + str(l)])
            dw = np.dot(dZ, parameters["A" + str(l - 1)].T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dAL = np.dot(parameters["W" + str(l)].T, dZ)
        parameters["W" + str(l)] = parameters["W" + str(l)] - (lr * dw)
        parameters["b" + str(l)] = parameters["b" + str(l)] - (lr * db)
if ivar==1 or ivar == len():
    if i>M:
        M=i
    i=0
def accuracy(X, Y, parameters, L):
    A = X
    for l in range(1,L-1):
        Z = np.dot(parameters["W"+str(l)], A)+(parameters["b"+str(l)])
        A = relu(Z)

    Z = np.dot(parameters["W"+str(L-1)], A)+(parameters["b"+str(L-1)])
    A = sigmoid(Z)
    prediction = np.zeros((1,m))
    for i in range(A.shape[1]):
        if A[:,i] >0.5:
            try:
                prediction[:,i] =1
            except:
                print("error")

    accc = 100 - np.mean(np.abs(prediction - Y)) * 100
    print(accc)
accuracy(A, Y, parameters, L)