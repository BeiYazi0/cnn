import numpy as np

from cnn.utils import NetworkDict, Kaiming_uniform


class Recurrent():
    __name__ = "Recurrent"
    '''
    循环层
    '''
    def __init__(self, units, input_shape, activate_fcn, theta = None):
        '''
        Args:
            units int: 输出的维度大小
            input_shape int: 输入该网络层的数据维度
            activate_fcn string: 激活函数
            theta (?): 指定的参数
        '''
        _dict = NetworkDict(0)
        if type(input_shape) != int:
            input_shape = input_shape[0]
        self.input_shape = (input_shape, )
        self.U_shape = (units, input_shape)
        self.W_shape = (units, units)
        # 初始化
        if theta is None or theta.shape != (units, input_shape + units + 1):
            self.U = Kaiming_uniform(input_shape, units)
            self.W = Kaiming_uniform(units, units)
            # 初始状态
            self.init_state = np.zeros((1, units))
        else:
            self.U = theta[:, :input_shape]
            self.W = theta[:, input_shape:-1]
            self.init_state = theta[:, -1]
        
        self.activate_fcn = _dict.get_activate_fcn(activate_fcn)
        self.activate_gradient_fcn = _dict.get_activate_gradient_fcn(activate_fcn)
        self.output_shape = (None, units)
        
        self._str = np.array([self.__name__, activate_fcn])
        
        # 输入，激活项
        self.x = None
        self.a = None
    
    def set_state(self, state):
        self.init_state = state
        
    def set_input(self, X):
        self.x = X
        
    def recurrent_forward(self, x, U, W, S_pre, activate_fcn):
        '''
        循环层输出
        Args:
            x (n, ): 输入
            U (t, n): 参数矩阵
            W (t, t): 参数矩阵
            S_pre (t, ): 上一时刻的状态
            activate_fcn method: 激活函数
        Returns:
            S_cur (t, ): 当前时刻的状态
        '''
        return activate_fcn(x @ U.T + S_pre @ W.T)
    
    def forward_propagate(self):
        batch_size = self.x.shape[0]
        units = self.output_shape[1]
        output = np.zeros((batch_size, units))
        
        # 修改本次前向传播的初始状态
        if self.a is not None:
            self.init_state = self.a[-1]
        
        cur_state = self.init_state # 初始状态
        for i in range(batch_size):
            cur_state = self.recurrent_forward(self.x[i], self.U, self.W, 
                                               cur_state, self.activate_fcn)
            output[i] = cur_state
        
        self.a = output
        return self.a
    
    def recurrent_backward(self, h, a, S_lst, error, U, W, activate_fcn_gradient):
        '''
        循环层系数更新和反向传播
        Args:
            h (m, t): 正向传播中循环层的激活项输出
            a (m, n): 正向传播中循环层的输入
            S_lst (m, t): 循环层的历史状态值
            error (m, t): error[i, :] 表示循环层在第 i 个时刻来自于下一层的误差
            U (t, n): 参数矩阵
            W (t, t): 参数矩阵
            activate_fcn_gradient method: 激活函数的梯度函数
        Returns:
            grad_U (t, n): 循环层系数 U 的梯度
            grad_W (t, t): 循环层系数 W 的梯度
            error_bp (m, n): 循环层向上一层反向传播的误差
        '''
        m, t = error.shape

        # 循环层每一个时刻的delta
        delta = np.zeros((m, t))
        v_gradient = activate_fcn_gradient(h)
        # 最后时刻的循环层误差仅来自于当前时刻的输出层
        error_bp_recurrent = np.zeros((1, t))
        for i in range(m - 1, -1, -1):
            # 当前时刻的循环层，其误差来自于当前时刻的输出层，以及下一时刻的循环层。
            t_error = error[i] + error_bp_recurrent
            delta[i] = np.multiply(t_error, v_gradient[i])
            error_bp_recurrent = delta[i] @ W # 传递给上一时刻循环层的误差

        # 计算 grad
        grad_U = delta.T @ a 
        grad_W = delta.T @ S_lst

        # 反向传播
        error_bp = delta @ U

        return grad_U, grad_W, error_bp
    
    def backward_propagate(self, error, lr):
        self.init_state = self.init_state.reshape(1, -1)
        S_lst = np.concatenate((self.init_state, self.a[:-1]), axis = 0)
        grad_U, grad_W, error_bp = self.recurrent_backward(self.a, self.x, S_lst, error,
                                                        self.U, self.W, self.activate_gradient_fcn)
        self.U -= lr * grad_U
        self.W -= lr * grad_W
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
        return self.__name__, self.output_shape, np.prod(self.U_shape, dtype=np.int32) + np.prod(self.W_shape, dtype=np.int32)
    
    def save(self):
        '''
        返回用于构建循环层的参数及参数矩阵的参数
        Args:
            None
        Returns:
            init_params (1, ): 构建隐层的参数
            params (input_shape, units): 隐层参数
            _str (2, ): 构建卷积层的参数
        '''
        init_params = np.array([self.U_shape[0]])
        if self.a is not None:
            self.init_state = self.a[-1]
        return init_params, np.concatenate((self.U, self.W, self.init_state.reshape(-1, 1)), axis = 1), self._str