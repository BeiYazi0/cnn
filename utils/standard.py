'''
本文件用于记录标准函数。
'''

import numpy as np
from scipy.signal import convolve2d


def conv_standard(x, kernel, padding = 0):
    '''
    卷积
    Args:
        x (N, C, H, W): 输入
        kernel (out_k, C, kH, kW): 卷积核
        padding int: 模式——0：valid; 1: same; 2: full.
    Returns:
        z (N, out_k, out_h, out_w): 卷积结果
    '''
    mode = ["valid", "same", "full"]
    mode = mode[padding]
    
    # 根据模式进行填充
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode = "constant")
    
    # 确定输出的大小
    N, C, H, W = x.shape
    out_k, C, kH, kW = kernel.shape
    out_h = (H + 2 * padding - kH) + 1
    out_w = (W + 2 * padding - kW) + 1
    
    # 卷积实现
    z = np.zeros((N, out_k, out_h, out_w))
    for i in range(N):
        for j in range(out_k):
            for ci in range(C):
                z[i, j] += convolve2d(x[i, ci], kernel[j, ci, ::-1, ::-1], 
                                      boundary = 'fill', mode = mode, fillvalue = 0)
    return z


def average_pool_standard(x, kernel_shape):
    '''
    平均池化
    Args:
        x (N, C, H, W): 输入
        kernel_shape tuple(int): 池化层参数
    Returns:
        z (N, C, out_h, out_w): 池化结果
    '''
    # 确定输出的大小
    N, C, H, W = x.shape
    kH, kW = kernel_shape
    out_h = H // kH
    out_w = W // kW
    
    # 平均池化
    z = np.zeros((N, C, out_h, out_w))
    for i in range(kH):
        for j in range(kW):
            z += x[:, :, i::kH, j::kW]
    
    return z/(kH * kW)


def max_pool_standard(x, kernel_shape):
    '''
    最大池化
    Args:
        x (N, C, H, W): 输入
        kernel_shape tuple(int): 池化层参数
    Returns:
        z (N, C, out_h, out_w): 池化结果
        max_id (N, C, out_h, out_w): 最大值神经元的 Max_ID位置
    '''
    # 确定输出的大小
    N, C, H, W = x.shape
    kH, kW = kernel_shape
    out_h = H // kH
    out_w = W // kW
    
    # 最大池化
    z = np.zeros((N, C, out_h, out_w))
    max_id = np.zeros((N, C, out_h, out_w), dtype = np.int32)
    for i in range(kH):
        for j in range(kW):
            target = x[:, :, i::kH, j::kW]
            mask = target > z
            max_id = max_id * (~mask) + mask * (i * kH + j)
            z = z * (~mask) + mask * target
    
    return z, max_id


def conv_bp_standard(x, z, error, kernel, activate_fcn_gradient):
    '''
    卷积层系数更新和反向传播
    Args:
        x (N, C, H, W): 正向传播中卷积层的输入
        z (N, out_k, out_h, out_w): 正向传播中卷积层的输出
        error (N, out_k, out_h, out_w): 从下一层反向传播而来的误差
        kernel (out_k, C, KH, KW): 卷积核
        activate_fcn_gradient method: 激活函数的梯度函数
        bp_flag boolean: 是否执行反向传播
    Returns:
        grad (out_k, C, KH, KW): 卷积层系数的梯度
        error_bp (N, C, H, W): 卷积层向上一层反向传播的误差
    '''
    N, C, H, W = x.shape
    out_k, C, KH, KW = kernel.shape
    
    # 计算delta
    delta = np.multiply(error, activate_fcn_gradient(z))
    
    # 计算 grad
    grad = np.zeros((out_k, C, KH, KW))
    for i in range(N):
        for j in range(out_k):
            for ci in range(C):
                grad[j, ci] += convolve2d(x[i, ci], delta[i, j][::-1, ::-1], mode = "valid")
    grad /= N
    
    # 反向传播
    error_bp = np.zeros((N, C, H, W))
    for i in range(N):
        for j in range(out_k):
            for ci in range(C):
                error_bp[i, ci] += convolve2d(delta[i, j], kernel[j, ci][::-1, ::-1], 
                                         boundary = 'fill', mode = "full", fillvalue = 0)
    
    return grad, error_bp


def average_pool_backward_standard(error, kernel_shape):
    '''
    平均池化的反向传播
    Args:
        error (N, out_k, out_h, out_w): 从下一层反向传播来的误差
        kernel_shape tuple(int): 池化核 (kH == KW)
    Returns:
        error_bp (N, out_k, KH * out_h, KW * out_w): 向上一层反向传播的误差
    '''
    KH, KW = kernel_shape
    # delta = error
    return error.repeat(KH, axis = -2).repeat(KW, axis = -1) / (KH * KW)


def max_pool_backward_standard(error, max_id, kernel_shape):
    '''
    最大池化的反向传播
    Args:
        error (N, out_k, out_h, out_w): 从下一层反向传播来的误差
        max_id (N, out_k, out_h, out_w): 最大值神经元的 Max_ID 位置
        kernel_shape tuple(int): 池化核 (kH == KW)
    Returns:
        error_bp (N, out_k, KH * out_h, KW * out_w): 向上一层反向传播的误差
    '''
    N, out_k, out_h, out_w = error.shape
    KH, KW = kernel_shape
    
    error_bp = np.zeros((N, out_k, KH * out_h, KW * out_w))
    for i in range(N):
        for j in range(out_k):
            row = max_id[i, j] // KH + np.arange(out_h).reshape(-1, 1) * KH
            col = max_id[i, j] % KH + np.arange(out_w).reshape(1, -1) * KW
            error_bp[i, j, row, col] = error[i, j]

    return error_bp