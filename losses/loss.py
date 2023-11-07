import numpy as np


def cross_tropy(h, y):
    '''
    交叉熵代价函数
    Args:
        h (m, k): 输出
        y (m, k): 真实值, k个类别
    Returns:
        cost float: 代价
    '''
    m = y.shape[0]
    
    # compute the cost
    J = np.multiply(-y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))
    cost = np.sum(J) / m

    return cost

def MSE(h, y):
    '''
    Mean squared error
    Args:
        h (m, k): 输出
        y (m, k): 真实值, k个类别
    Returns:
        cost float: 代价
    '''
    m = y.shape[0]
    
    # compute the cost
    J = np.power(h - y, 2)
    cost = np.sum(J)/2/m

    return cost