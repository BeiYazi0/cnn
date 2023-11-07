import numpy as np
import math


def Kaiming_uniform(fan_in, fan_out, a = math.sqrt(5)):
    '''
    Kaiming 均匀分布的初始化
    Args:
        fan_in int: 输入神经元的数量
        fan_out int: 输出神经元的数量
        a float: Relu 或 Leaky Relu的负半轴斜率，用来衡量这一层中负数比例
    '''
    bound = 6.0 / (1 + a * a) / fan_in
    bound = math.sqrt(bound)
    return np.random.uniform(low=-bound, high=bound, size=(fan_out, fan_in))


def Kaiming_std(fan_in, target_shape, a = math.sqrt(5)):
    '''
    Kaiming 正态分布的初始化
    Args:
        fan_in int: 输入神经元的数量
        target_shape (out_k, C, H, W): 目标参数矩阵shape
        a float: Relu 或 Leaky Relu的负半轴斜率，用来衡量这一层中负数比例
    '''
    bound = 2.0 / (1 + a * a) / fan_in
    std = math.sqrt(bound)
    return np.random.normal(loc = 0, scale = std, size = target_shape)