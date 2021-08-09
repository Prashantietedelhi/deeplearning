import numpy as np
n_x = 12288    # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    parameters = {}
    parameters["W1"] = np.random.randn(n_h, n_x)*0.01
    parameters["b1"] = np.zeros((n_h,1))
    parameters["W2"] = np.random.randn(n_y, n_h)*0.01
    parameters["b2"] = np.zeros((n_y,1))
    return parameters
def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    c  = 1/(1+np.exp(-Z))
    return c

def linear_activation_forward(X, W, b, activation):
    Z = np.dot(W,X)+b
    if activation =="relu":
        A = relu(Z)
    elif activation == 'sigmoid':
        A = sigmoid(Z)
    return A,Z

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))) / m
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost

def der_output(A, Y):
    return -np.divide(Y,A) - np.divide(1-Y, 1-A)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    return dZ
def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def linear_activation_backward(dA, W, Z, A_prev, activation):
    if activation == "relu":
        dZ = relu_backward(dA,Z)
        dW = np.dot(dZ, A_prev.T)/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        dA_prev = np.dot(W.T, dZ)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,Z)
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        dA_prev = np.dot(W.T, dZ)
    return dA_prev,dW, db

def update_parameters(parameters , grads, learning_rate):
    # print(grads['db1'])
    parameters["W1"] = parameters["W1"] - (learning_rate*grads['dW1'])
    parameters["W2"] = parameters["W2"] - (learning_rate*grads['dW2'])
    parameters["b1"] = parameters["b1"] - (learning_rate*grads['db1'])
    parameters["b2"] = parameters["b2"] - (learning_rate*grads['db2'])
    return parameters

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
train_set_x_flatten = train_set_x_flatten/255.
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_flatten/255.
X = train_set_x_flatten
Y= train_set_y_orig

# X = np.array([[1.,2.,-1.],[3.,4.,-3.2]])
# Y = np.array([[1,0,1]])

grads = {}
costs = []  # to keep track of the cost
m = X.shape[1]  # number of examples
(n_x, n_h, n_y) = layers_dims

# Initialize parameters dictionary, by calling one of the functions you'd previously implemented
### START CODE HERE ### (≈ 1 line of code)
parameters = initialize_parameters(n_x, n_h, n_y)
### END CODE HERE ###

# Get W1, b1, W2 and b2 from the dictionary parameters.
W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]

# Loop (gradient descent)
num_iterations = 20
learning_rate = 0.055
for i in range(0, num_iterations):
    # X = np.array([[1., 2., -1.], [3., 4., -3.2]])
    # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1,Z1 = linear_activation_forward(X, W1, b1, 'relu')
    A2,Z2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
    ### END CODE HERE ###

    # Compute cost
    ### START CODE HERE ### (≈ 1 line of code)
    cost = compute_cost(A2, Y)
    ### END CODE HERE ###
    print(cost)
    # Initializing backward propagation
    dA2 = der_output(A2,Y)

    # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
    ### START CODE HERE ### (≈ 2 lines of code)
    dA1, dW2, db2 = linear_activation_backward(dA2, W2, Z2, A1, 'sigmoid')
    dA0, dW1, db1 = linear_activation_backward(dA1, W1, Z1, X, 'relu')
    ### END CODE HERE ###

    # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
    grads['dW1'] = dW1
    grads['db1'] = db1
    grads['dW2'] = dW2
    grads['db2'] = db2

    # Update parameters.
    ### START CODE HERE ### (approx. 1 line of code)
    parameters = update_parameters(parameters, grads, learning_rate)
    ### END CODE HERE ###

    # Retrieve W1, b1, W2, b2 from parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Print the cost every 100 training example

# plot the cost





