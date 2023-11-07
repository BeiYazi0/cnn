import numpy as np

def get_im2col_indices(x_shape, field_height, field_width, padding=0, stride=1):
    '''
    im2col 的高效实现
    Args:
        x_shape tuple(int): 样本数、图像通道数、图像高度、图像宽度
        field_height int: 卷积核高
        field_width int: 卷积核宽
        padding int: 模式——0：valid; ?: same; ?: full.
        stride int: 步长
    Returns:
        tuple((field_height*field_width*C, 1),
        (field_height*field_width*C, out_height*out_width),
        (field_height*field_width*C, out_height*out_width)): 索引
    '''
    N, C, H, W = x_shape
    
    # 卷积输出的大小
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    ## 生成行索引
    # 卷积核框住区域中，所有元素的行索引相对被框住区域左上角元素的偏移量（从左到右，从上到下）
    i0 = np.repeat(np.arange(field_height), field_width) 
    # 复制i0，次数为通道数
    i0 = np.tile(i0, C)
    # 卷积运算时，每个框选区域的左上角元素的行索引
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    
    ## 生成列索引
    # 卷积核框住区域中，所有元素的列索引相对被框住区域左上角元素的偏移量（从左到右，从上到下）， 复制C次
    j0 = np.tile(np.arange(field_width), field_height * C)
    # 卷积运算时，每个框选区域的左上角元素的列索引
    j1 = stride * np.tile(np.arange(out_width), out_height)
    
    ## 生成行列的二维索引
    i = i0.reshape(-1, 1) + i1.reshape(1, -1) # i[m, :]表示参加第m次卷积运算的所有元素的行索引
    j = j0.reshape(-1, 1) + j1.reshape(1, -1) # j[m, :]表示参加第m次卷积运算的所有元素的列索引

    ## 生成通道数索引
    # 卷积核框住区域中，所有元素的通道数索引相对被框住区域左上角元素的偏移量（从左到右，从上到下，从通道0到C）
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    '''
    im2col
    Args:
        x (N, C, H, W): 输入
        field_height int: 卷积核高
        field_width int: 卷积核宽
        padding int: 模式——0：valid; ?: same; ?: full.
        stride int: 步长
    Returns:
        cols (field_height*field_width*C, out_height*out_width*N): 与卷积核相乘实现卷积的的矩阵
    '''
    # 根据模式进行填充
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")

    # 获取cols对应的索引值
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    # 满足 W.T @ cols = output
    return cols.transpose(1, 0, 2).reshape(field_height * field_width * C, -1)

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=0, stride=1):
    '''
    col2im，将 col 还原
    Args:
        cols (field_height*field_width*C, out_height*out_width*N): 与卷积核相乘实现卷积的的矩阵
        x_shape tuple(int): 样本数、图像通道数、图像高度、图像宽度
        field_height int: 卷积核高
        field_width int: 卷积核宽
        padding int: 模式——0：valid; ?: same; ?: full.
        stride int: 步长
    Returns:
        x (N, C, H_padded, W_padded): 原输入
    '''
    
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    
    # 获取cols对应的索引值
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]