import numpy as np
import matplotlib.pylab as plt
from ch4.loss_function import numerical_diff

def numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # create array (same size as x)

    for idx in range(x.size):
        # cal f(x+h)
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # cal f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # restore the value

    return grad

def numerical_gradient(f, X):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(X)

    it = np.nditer(X, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = X[idx]
        X[idx] = float(tmp_val) + h
        fxh1 = f(X)  # f(x+h)

        X[idx] = tmp_val - h
        fxh2 = f(X)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        X[idx] = tmp_val  # restore value
        it.iternext()

    return grad

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)

    X = X.flatten()
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666", headwidth=10, scale=40)
    # We use quiver when we want to draw scaled arrow at each point
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()
