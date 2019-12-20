import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

x = np.random.randn(1000, 100) # 1000 input data
node_num = 100 # # of node(neuron) per each hidden layer
hidden_layer_size = 5 # # of hidden layer : 5
activations = {} # store activation result

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # initial weight value
    # w = np.random.randn(node_num, node_num) * 1 # multiply 'std'
    # w = np.random.randn(node_num, node_num) * 0.01 # restrict expression
    w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)

    a = np.dot(x, w)

    # activation function
    # z = sigmoid(a)
    # z = ReLU(a)
    z = tanh(a)


    activations[i] = z

# draw histogram
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()