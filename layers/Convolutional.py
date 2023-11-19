import numpy as np

from cnn.utils import NetworkDict, im2col_indices, Kaiming_std


class Conv2D():
    __name__ = "Conv2D"
    '''
    卷积层
    当前仅用于padding = valid, stride = 1
    '''
    def __init__(self, filters, kernel_size, stride = 1,
                 input_shape = None, padding = "valid", activate_fcn = "ReLU",
                 kernel = None):
        '''
        Args:
            filters int: 卷积核个数
            kernel_size int: 卷积核大小
            input_shape (C, H, W): 该层输入的维度
            stride int: 步长
            padding string: valid; same; full.
            activate_fcn string: 激活函数
            kernel (?): 指定的卷积核参数
        '''
        _dict = NetworkDict(kernel_size)
        C, H, W = input_shape
        self.kernel_shape = (filters, input_shape[0], kernel_size, kernel_size)
        if kernel is None or kernel.shape != self.kernel_shape:
            self.kernel = Kaiming_std(np.prod(input_shape), self.kernel_shape)
        else:
            self.kernel = kernel
        
        self.activate_fcn = _dict.get_activate_fcn(activate_fcn)
        self.activate_gradient_fcn = _dict.get_activate_gradient_fcn(activate_fcn)
        
        self.input_shape = input_shape
        self.stride = stride
        self.padding = _dict.get_padding(padding)
        self.output_shape = (None, filters, (H + 2 * self.padding - kernel_size) // stride + 1, (W + 2 * self.padding - kernel_size) // stride + 1)
        
        self._str = np.array([self.__name__, padding, activate_fcn])
        
        # 输入，激活项
        self.x = None
        self.a = None
    
    def set_input(self, X):
        self.x = X
        
    def conv_im2col(self, x, kernel, padding = 0, stride = 1):
        '''
        卷积的高效实现
        Args:
            x (N, C, H, W): 输入
            kernel (out_k, C, kH, kW): 卷积核
            padding int: 模式——0：valid; 1: same; 2: full.
            stride int: 步长
        Returns:
            z (N, out_k, out_h, out_w): 卷积结果
        '''
        N, C, H, W = x.shape
        out_k, C, kH, kW = kernel.shape
        out_h = (H + 2 * padding - kH) // stride + 1
        out_w = (W + 2 * padding - kW) // stride + 1

        # im2col
        x_col = im2col_indices(x, kH, kW, padding, stride)
        kernel_col = kernel.reshape(out_k, -1)

        # 卷积实现
        z = kernel_col @ x_col

        return z.reshape(out_k, N, out_h, out_w).transpose(1, 0, 2, 3)
    
    def forward_propagate(self):
        z = self.conv_im2col(self.x, self.kernel, self.padding, self.stride)
        self.a = self.activate_fcn(z)
        return self.a
    
    def conv_bp(self, x, a, error, kernel, activate_fcn_gradient):
        '''
        卷积层系数更新和反向传播
        Args:
            x (N, C, H, W): 正向传播中卷积层的输入
            a (N, out_k, out_h, out_w): 正向传播中卷积层的激活输出
            error (N, out_k, out_h, out_w): 从下一层反向传播而来的误差
            kernel (out_k, C, KH, KW): 卷积核
            activate_fcn_gradient method: 激活函数的梯度函数
            bp_flag boolean: 是否执行反向传播
        Returns:
            grad (out_k, C, KH, KW): 卷积层系数的梯度
            error_bp (N, C, H, W): 卷积层向上一层反向传播的误差
        '''
        N, C, H, W = x.shape

        # 计算delta
        delta = np.multiply(error, activate_fcn_gradient(a))

        # 计算 grad
        grad = self.conv_im2col(x.transpose(1, 0, 2, 3), 
                           delta.transpose(1, 0, 2, 3)).transpose(1, 0, 2, 3) / N

        # 反向传播
        error_bp = self.conv_im2col(delta, kernel.transpose(1, 0, 2, 3), 
                                    padding = kernel.shape[-1]-1)

        return grad, error_bp
    
    def backward_propagate(self, error, lr):
        grad, error_bp = self.conv_bp(self.x, self.a, error, 
                                      self.kernel, self.activate_gradient_fcn)
        self.kernel -= lr * grad
        return error_bp
    
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
        return self.__name__, self.output_shape, np.prod(self.kernel_shape, dtype=np.int32)
    
    def save(self):
        '''
        返回用于构建卷积层的参数及卷积核的参数
        Args:
            None
        Returns:
            init_params (3, ): 构建卷积层的参数
            params (filters, input_shape[0], kernel_size, kernel_size): 卷积核参数
            _str (3, ): 构建卷积层的参数
        '''
        init_params = np.array([self.kernel_shape[0], self.kernel_shape[-1], self.stride])
        return init_params, self.kernel, self._str