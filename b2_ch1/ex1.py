import numpy as np
from b2_common.function import sigmoid

x = np.random.randn(10, 2) # shape (10, 2)
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4) # broadcast
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.matmul(x, W1) + b1
a = sigmoid(h)
s = np.matmul(a, W2) + b2 # shape (10, 3)


