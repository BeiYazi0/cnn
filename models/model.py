import numpy as np

from cnn.losses import *
from .modelSave import save_model


class Model():
    '''
    神经网络
    '''
    def __init__(self, Input_layer, name = "cnn"):
        '''
        Args:
            Input_layer <class 'Input'>: 输入层
            name string: Model名
        Returns:
            None
        '''
        self.input = Input_layer
        self.cur_output_shape = Input_layer.input_shape
        self.name = name
        
        self.layers = [] # 网络层
        self.lr = 0.0    # 学习率
        
        self.loss_fcn = None     # 损失函数
        self.loss_fcn_name = "_" # 损失函数名
    
    def add_layer(self, layer):
        '''
        添加层
        Args:
            layer <class '?'>: 网络层
        Returns:
            None
        '''
        assert layer.input_shape == self.cur_output_shape
        self.layers.append(layer)
        self.cur_output_shape = layer.output_shape[1:]
    
    def compile(self, learning_rate, loss_fcn = "cross_tropy"):
        '''
        设置损失函数 loss、学习率 learning_rate
        Args:
            learning_rate float: 学习率
            loss_fcn string: 损失函数
        Returns:
            None
        '''
        assert learning_rate > 1e-6 and learning_rate < 1
        self.lr = learning_rate
        self.loss_fcn_name = loss_fcn
        
        loss_dic = {"cross_tropy": cross_tropy}
        self.loss_fcn = loss_dic[loss_fcn] if loss_fcn in loss_dic else MSE
            
    def forward(self):
        '''
        前向传播
        Args:
            None
        Returns:
            a (m, k): 输出
        '''
        a = self.input.fordwrd_propagate()
        for layer in self.layers:
            layer.set_input(a)
            a = layer.fordwrd_propagate()
        return a
    
    def backward(self, error):
        '''
        反向传播
        Args:
            error (N, k): 误差
        Returns:
            None
        '''
        for layer in self.layers[::-1]:
            error = layer.backward_propagate(error, self.lr)
    
    def fit(self, x, y, batch_size = -1, epochs = 1, verbose = 1, shuffle = True):
            '''
            训练模型
            Args:
                x (N, C, H, W): 输入
                y (N, k): 输出
                batch_size int: 每次梯度更新的样本数
                epochs int: 训练模型迭代次数
                verbose int: 日志展示
                    0:不在标准输出流输出日志信息
                    1:显示进度条
                    2:每个epoch输出一行记录
                shuffle boolean: 是否在每轮迭代之前混洗数据
            Returns:
                history dict{string: (epochs, )}: 准确率和损失历史值
            '''
            N = x.shape[0]                         # 样本数量
            batchs = int(np.ceil(N / batch_size))  # 总 batch 数
            index = np.arange(N)                   # 用于随机打乱的索引
            y_true = y.argmax(axis=1)              # label

            # 默认为批量
            if batch_size == -1:
                batch_size = N

            history = {"accuracy": np.zeros((epochs)), "loss": np.zeros((epochs))}
            print("Model train start.")
            print("=================================================================")
            for i in range(epochs):
                if shuffle: # 每轮 epoch 打乱数据
                    np.random.shuffle(index)
                    x = x[index]
                    y = y[index]
                    y_true = y.argmax(axis=1)
                h = np.zeros(y.shape) # 每轮的输出
                for j in range(0, N, batch_size):
                    k = min(j+batch_size, N)
                    Xs = x[j:k] # 每次取 batch_size 个数据
                    ys = y[j:k]
                    self.input.set_input(Xs)

                    # 前向传播
                    a = self.forward()
                    h[j:k] = a

                    if verbose == 1: # batch 日志
                        accuracy = np.sum(y_true[j:k] == a.argmax(axis=1)) / (k - j)
                        print("batch %8s/%-8s\taccuracy: %-10s\tloss: %-10s" % (j//batch_size+1, batchs, np.round(accuracy, 6), np.round(self.loss_fcn(a, ys), 6)))

                    # 后向传播
                    self.backward(a - ys)
                    
                history["loss"][i] = self.loss_fcn(h, y)
                history["accuracy"][i] = np.sum(y_true == h.argmax(axis=1)) / N
                if verbose > 0: # epoch 日志
                    print("_________________________________________________________________")
                    print("epoch %8s/%-8s\taccuracy: %-10s\tloss: %-10s" % (i+1, epochs, np.round(history["accuracy"][i], 6), np.round(history["loss"][i], 6)))
                    print("=================================================================")
            return history
    
    def predict(self, test_data):
        '''
        预测输出
        Args:
            test_data (m, n): 输入
        Return:
            a (m, k): 输出
        '''
        self.input.set_input(test_data)
        return self.forward()
    
    def predict_classes(self, test_data):
        '''
        预测分类
        Args:
            test_data (m, n): 输入
        Return:
            classes (m, 1): 输出
        '''
        return self.predict(test_data).argmax(axis = 1)
    
    def evaluate(self, x_test, y_test):
        '''
        模型在测试数据上的准确率和损失
        Args:
            x_test (m, n): 输入
            y_test (m, k): label
        Return:
            accuracy float: 准确率
            loss float: 损失
        '''
        a = self.predict(x_test)
        return np.sum(y_test.argmax(axis = 1) == a.argmax(axis = 1)) / a.shape[0], self.loss_fcn(a, y_test)
    
    def summary(self):
        '''
        查看模型各个层的组成
        Args:
            None
        Returns:
            None
        '''
        total_params = 0
        print("model name: " + self.name)
        print("_________________________________________________________________")
        print("Layer                        Output Shape              Param #   ")
        print("=================================================================")
        for layer in self.layers:
            name, input_shape, params = layer.summary()
            total_params += params
            print("%-29s%-26s%-28s" % (name, input_shape, params))
            print("_________________________________________________________________")
        print("=================================================================")
        print("Total params: %d" % total_params)
        print("_________________________________________________________________")
        
    def save(self, filename):
        '''
        保存模型
        Args:
            filename string: 文件名
        Returns:
            None
        '''
        save_model(filename, self)