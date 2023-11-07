from cnn.activations import *

class NetworkDict():
    def __init__(self, size):
        self.activate_fcn_dic = {"ReLU": ReLU, "sigmoid": sigmoid, "softmax": softmax}
        self.activate_gradient_fcn_dic = {"ReLU": ReLU_gradient, "sigmoid": sigmoid_gradient, 
                                     "softmax": softmax_gradient}
        self.padding_dic = {"valid": 0, "same": size//2, "full": size-1}
        
    def get_activate_fcn(self, activate_fcn):
        if activate_fcn in self.activate_fcn_dic:
            return self.activate_fcn_dic[activate_fcn]
        return Linear
    
    def get_activate_gradient_fcn(self, activate_fcn):
        if activate_fcn in self.activate_gradient_fcn_dic:
            return self.activate_gradient_fcn_dic[activate_fcn]
        return Linear_gradient
    
    def get_padding(self, padding):
        assert padding in self.padding_dic
        return self.padding_dic[padding]