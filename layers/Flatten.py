import numpy as np


class Flatten():
    __name__ = "Flatten"
    '''
    flatten 层
    把多维的输入一维化
    '''
    def __init__(self, input_shape):
        '''
        Args:
            input_shape (C, H, W): 处理图像的 shape
        '''
        self.input_shape = input_shape
        self.output_shape = (None, np.prod(input_shape))
        
        self._str = np.array([self.__name__])
        
        # 输入，输出
        self.x = None
        self.z = None
    
    def set_input(self, X):
        self.x = X
        
    def flatten_forward(self, x):
        '''
        flatten 
        Args:
            x (N, C, H, W): 输入
        Returns:
            z (N, C*H*W): flatten 输出
        '''
        N, C, H, W = x.shape
        self.input_shape = (C, H, W)
        return x.reshape(N, -1)
    
    def forward_propagate(self):
        self.z = self.flatten_forward(self.x)
        return self.z
    
    def flatten_backford(self, error, input_shape):
        '''
        flatten reverse
        Args:
            error (N, -1): 从下一层反向传播来的误差
            input_shape (C, H, W): 输入 flatten 层前的形状
        Returns:
            error_bp (N, C, H, W): 向上一次反向传播的误差
        '''
        C, H, W = input_shape

        return error.reshape((error.shape[0], C, H, W))
    
    def backward_propagate(self, error, lr):
        return self.flatten_backford(error, self.input_shape)
    
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
        返回用于构建卷积层的参数及卷积核的参数
        Args:
            None
        Returns:
            init_params (?, ): 构建卷积层的参数
            params (?): 参数
            _str (2, ): 构建卷积层的参数
        '''
        return None, None, self._str