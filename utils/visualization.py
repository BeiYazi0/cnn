import numpy as np
import itertools
from matplotlib import pyplot as plt


def history_show(history, accuracy_file = None, loss_file = None):
    # accuracy的历史
    plt.plot(history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    if accuracy_file is not None:
        plt.savefig(accuracy_file)
    plt.show()
    # loss的历史
    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    if loss_file is not None:
        plt.savefig(loss_file)
    plt.show()

def confusion_show(labels, y_pred, y_true, normalize = False, confusion_file = None):
    '''
    混淆矩阵可视化
    Args:
        labels List[string]: 标签
        y_pred (m, 1): 预测分类
        y_true (m, 1): 真实分类
        normalize boolean: 归一化
        confusion_file string: 文件名
    Returns:
        None
    '''
    classes = len(labels) # 总类别数
    
    # 混淆矩阵
    cm = np.bincount(classes * y_true.astype(np.int32) + y_pred, 
                minlength = classes**2).reshape(classes, classes) 
    if normalize:
        cm = cm.astype(np.float64) / cm.max()
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('confusion matrix')
    plt.colorbar()
    
    plt.xticks(range(classes), labels, rotation=45)
    plt.yticks(range(classes), labels)
    plt.ylim(classes - 0.5, -0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if confusion_file is not None:
        plt.savefig(confusion_file)
    plt.show()