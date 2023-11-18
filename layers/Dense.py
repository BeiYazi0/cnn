import numpy as np

from cnn.utils import NetworkDict, Kaiming_uniform


class Dense():
    __name__ = "Dense"
    '''
    全连接层
    '''
    def __init__(self, units, input_shape, activate_fcn, theta = None):
        '''
        Args:
            units int: 输出的维度大小
            input_shape int: 输入该网络层的数据
            activate_fcn string: 激活函数
            theta (?): 指定的参数
        '''
        _dict = NetworkDict(0)
        if type(input_shape) != int:
            input_shape = input_shape[0]
        self.input_shape = (input_shape, )
        self.theta_shape = (units, input_shape)
        if theta is None or theta.shape != self.theta_shape:
            self.theta = Kaiming_uniform(input_shape, units)
        else:
            self.theta = theta
        
        self.activate_fcn = _dict.get_activate_fcn(activate_fcn)
        self.activate_gradient_fcn = _dict.get_activate_gradient_fcn(activate_fcn)
        self.output_shape = (None, units)
        
        self._str = np.array([self.__name__, activate_fcn])
        
        # 输入，激活项
        self.x = None
        self.a = None
    
    def set_input(self, X):
        self.x = X
        
    def hidden_forward(self, x, theta, activate_fcn):
        '''
        hidden
        Args:
            x (m, n): m 个样本，n个特征
            theta (t, n): t 个输出神经元，n个输入神经元
            activate_fcn method: 激活函数
        Returns:
            a (m, t): 激活输出
        '''
        z = x @ theta.T
        a = activate_fcn(z)

        return a
    
    def fordwrd_propagate(self):
        self.a = self.hidden_forward(self.x, self.theta, self.activate_fcn)
        return self.a
    
    def hidden_backward(self, a, x, error, theta, activate_fcn_gradient):
        '''
        隐层系数更新和反向传播
        Args:
            a (m, t): 正向传播中隐层的激活输出
            x (m, n): 正向传播中隐层的输入
            error (m, t): 从下一层反向传播而来的误差
            theta (t, n): 参数矩阵
            activate_fcn_gradient method: 激活函数的梯度函数
        Returns:
            grad (n, t): 隐层系数的梯度
            error_bp (m, n): 隐层向上一层反向传播的误差
        '''
        # 计算delta
        delta = np.multiply(error, activate_fcn_gradient(a))

        # 计算 grad
        grad = delta.T @ x 

        # 反向传播
        error_bp = delta @ theta

        return grad, error_bp
    
    def backward_propagate(self, error, lr):
        grad, error_bp = self.hidden_backward(self.a, self.x, error, 
                                              self.theta, self.activate_gradient_fcn)
        self.theta -= lr * grad
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
        return self.__name__, self.output_shape, np.prod(self.theta_shape, dtype=np.int32)
    
    def save(self):
        '''
        返回用于构建隐层的参数及卷积核的参数
        Args:
            None
        Returns:
            init_params (1, ): 构建隐层的参数
            params (input_shape, units): 隐层参数
            _str (2, ): 构建卷积层的参数
        '''
        init_params = np.array([self.theta_shape[0]])
        return init_params, self.theta, self._str