class Input():
    __name__ = "Input"
    '''
    输入层
    '''
    def __init__(self, input_shape):
        '''
        Args:
            input_shape (C, H, W): 输入该网络层的数据
        '''
        self.input_shape = input_shape
        
        # 输入
        self.x = None
    
    def set_input(self, X):
        self.x = X
    
    def fordwrd_propagate(self):
        return self.x
    
    def summary(self):
        '''
        返回层类型、输入数据维度、参数量
        Args:
            None
        Returns:
            __name__ string: 层类型
            input_shape tuple(int): 输入数据维度
            params int: 参数量
        '''
        return self.__name__, self.input_shape, np.sum(self.input_shape, dtype=np.int32)