import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('C:\\Users\\coole\\Documents\\AI\\MNIST\\Resources\\mnist_train.csv')
#data = pd.read_csv('C:\\Users\\coole\\Documents\\AI\\MNIST\\Resources\\mnist_test.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

#M goes down the array tracking number of ROWS
#N goes across the array tracking number of COLUMNS

#Across each row is the data stored for 1 image, with its actual value and pixel value for each of the 784 pixels

data = data[0:10].T
Y = data[0]
#Dividing by 255 to clamp pixel values between 0 and 1
X = data[1:n]/255

#Y is Labels
#X is Pixel Values

#Creating random Weights and Biases
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

W1, b1, W2, b2 = init_params()

def Leaky_ReLU(Layer):
    return np.maximum(0.1*Layer,Layer)

def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z))
    return A

#Feeding data through the network and modifying it with Weights and Biases
def forward_propagation(W1, b1, W2, b2, X):
    #H1 is Hidden Layer 1 and X is input Layer
    L1 = W1.dot(X) + b1
    #Activation Layer
    A1 = Leaky_ReLU(L1)
    #Output Layer
    L2 = W2.dot(A1) + b2
    #Activation Layer
    A2 = softmax(L2)

    return L1, A1, L2, A2
