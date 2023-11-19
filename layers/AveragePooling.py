import numpy as np

from cnn.utils import NetworkDict, im2col_indices


class AveragePooling2D():
    __name__ = "AveragePooling2D"
    '''
    平均池化层
    当前仅用于padding = valid, stride = pool_size
    '''
    def __init__(self, pool_size, stride = 0, input_shape = None, padding='valid'):
        '''
        Args:
            pool_size int: 池化核大小
            input_shape (C, H, W): 处理图像的 shape
            stride int: 步长
            padding string: valid; same; full.
        '''
        _dict = NetworkDict(pool_size)
        C, H, W = input_shape
        self.input_shape = input_shape
        self.pool_shape = (pool_size, pool_size) 
        self.stride = stride if stride > 0 else pool_size
        self.padding = _dict.get_padding(padding)
        self.output_shape = (None, C, H // self.stride, W // self.stride)
        
        self._str = np.array([self.__name__, padding])
        
        # 输入，输出
        self.x = None
        self.z = None
    
    def set_input(self, X):
        self.x = X
        
    def average_pool_im2col(self, x, kernel_shape, padding = 0, stride = 0):
        '''
        平均池化的高效实现
        Args:
            x (N, C, H, W): 输入
            kernel (kH, kW): 池化核 (kH == KW)
            stride int: 步长
            padding string: valid; same; full. #未使用（为valid）
        Returns:
            z (N, out_k, out_h, out_w): 卷积结果
        '''
        N, C, H, W = x.shape
        kH, kW = kernel_shape
        if stride == 0:
            stride = KH
        out_h = H // stride
        out_w = W // stride

        # im2col
        x_col = im2col_indices(x, kH, kW, 0, stride).reshape(C, kH*kW, -1)

        # 平均池化
        z = x_col.mean(axis = 1)

        return z.reshape(C, N, out_h, out_w).transpose(1, 0, 2, 3)

    
    def forward_propagate(self):
        self.z = self.average_pool_im2col(self.x, self.pool_shape, 
                                                   self.padding, self.stride)
        return self.z
    
    def average_pool_backward(self, error, kernel_shape):
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
    
    def backward_propagate(self, error, lr):
        return self.average_pool_backward(error, self.pool_shape)
    
    def summary(self):
        '''
        返回层类型、输入数据维度、参数量
        Args:
            None
        Returns:
            __name__ string: 层类型
            output_shape tuple(int): 输出数据维度
            params int: 参数量
        '''
        return self.__name__, self.output_shape, 0
    
    def save(self):
        '''
        返回用于构建池化层的参数及卷积核的参数
        Args:
            None
        Returns:
            init_params (?, ): 构建池化层的参数
            params (?): 参数
            _str (2, ): 构建卷积层的参数
        '''
        init_params = np.array([self.pool_shape[0], self.stride])
        return init_params, None, self._str