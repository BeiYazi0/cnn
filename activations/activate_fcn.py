import numpy as np


def ReLU(z):
    '''
    ReLU 函数
    Args:
        z (m, n): 输入
    Returns:
        g (m, n): ReLU 函数输出
    '''
    return np.maximum(0, z)

def sigmoid(z):
    '''
    sigmoid 函数
    Args:
        z (m, n): 输入
    Returns:
        g (m, n): sigmoid 函数输出
    '''
    d = 1 + np.exp(-z)
    return 1. / d

def softmax(z):
    '''
    softmax 函数
    Args:
        z (m, n): 输入
    Returns:
        g (m, n): softmax 函数输出
    '''
    d = np.exp(z)
    # 注意 d sum时的axis
    return d / d.sum(axis = 1).reshape(-1, 1)

def tanh(z):
    '''
    tanh 函数
    Args:
        z (m, n): 输入
    Returns:
        g (m, n): tanh 函数输出
    '''
    b = np.exp(z)
    c = np.exp(-z)
    return (b - c) / (b + c)

def Linear(z):
    '''
    Linear 函数
    Args:
        z (m, n): 输入
    Returns:
        g (m, n): Linear 函数输出
    '''
    return z

def ReLU_gradient(h):
    return h > 0

def sigmoid_gradient(h):
    return np.multiply(h, (1 - h))

def softmax_gradient(h):
    return np.multiply(h, (1 - h))

def tanh_gradient(h):
    return 1 - np.power(h, 2)

def Linear_gradient(z):
    return 1