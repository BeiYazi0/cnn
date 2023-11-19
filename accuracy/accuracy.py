import numpy as np


def MAE(y, h):
    '''
    平均绝对误差，用于回归问题
    Args:
        y [m, n]: 真实值
        h [m, n]: 预测输出
    '''
    AE = np.abs(h - y)
    MAE = np.sum(AE.mean(axis = 0))
    return "MAE: %-10s" % np.round(MAE, 6), MAE


def categorical_accuracy(y, h):
    '''
    categorical_accuracy，用于分类问题
    Args:
        y [m, n]: 真实值(one-hot标签)
        h [m, n]: 预测输出
    '''
    accuracy = np.sum(y.argmax(axis=1) == h.argmax(axis=1)) / y.shape[0]
    return "accuracy: %-10s" % np.round(accuracy, 6), accuracy