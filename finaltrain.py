# coding=utf-8
import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
# helper to load data from PNG image files
import imageio
import PIL
import json


# 保存对象到文件
def save_object(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj.__dict__, f)

# 从文件加载对象
def load_object(filename, cls):
    with open(filename, 'r') as f:
        data = json.load(f)
    obj = cls.__new__(cls)
    obj.__dict__.update(data)
    return obj

# neural network class definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc

        # 随机取即可,这个是一个经验函数
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        # sigmoid 函数，它能将任意输入压缩到 0 和 1 之间，使得神经元的输出可以解释为概率
        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        # 将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        # self.wih 是从输入层到隐藏层的权重矩阵
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        """
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))：
        这行代码首先计算了输出层的误差(output_errors)与输出层输出(final_outputs)的梯度的乘积
        这个梯度实际上就是 sigmoid 函数的导数 (final_outputs * (1.0 - final_outputs))。
        然后，这个结果与隐藏层的输出(hidden_outputs)的转置进行点乘。最后，将这个结果乘以学习率(self.lr)，
        然后累加到现有的输出层到隐藏层的权重(self.who)上。
        """

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        """self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))：
        这行代码的计算过程与上面的类似，只不过这里是计算隐藏层的误差(hidden_errors)与隐藏层输出(hidden_outputs)的梯度的乘积，
        然后这个结果与输入层的输出(inputs)的转置进行点乘，
        最后将这个结果乘以学习率(self.lr)，
        然后累加到现有的输入层到隐藏层的权重(self.wih)上。
        """
        pass

    # query the neural network
    """
    rain函数和query函数的主要关系在于，train函数是用来训练并优化神经网络的，
    而query函数则是用来利用已经训练好的网络进行预测的。
    在神经网络训练好之后，我们就可以使用query函数来获取网络对新输入数据的预测结果。
    """

    def query(self, inputs_list):
        # 输入列表转换为 numpy 数组，然后将其转置
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# number of input, hidden and output nodes

"""
input_nodes = 784:定义了神经网络的输入节点数。
在这里，输入节点数为784，这可能是因为在处理图像数据时，图像大小被重设为28x28像素，
因此总共有784个像素，每个像素对应一个输入节点。

hidden_nodes = 200:定义了神经网络的隐藏节点数。隐藏层的节点数量可以根据问题的复杂性和训练数据的量来设定。
一般来说，更复杂的问题和更大的数据集可能需要更多的隐藏节点。

output_nodes = 10：定义了神经网络的输出节点数。
在这里，输出节点数为10，这可能是因为该神经网络用于处理一个10类的分类问题（例如，手写数字识别，有10个数字从0到9）。

learning_rate = 0.1：定义了神经网络的学习率。学习率是一个超参数，用于控制神经网络权重更新的步长。它的设置对神经网络的训练效果有重要影响。
"""

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open(r'C:\Users\User\Desktop\MNIST识别\mnist_test.csv')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 2  # 设置迭代次数
print("start")
i=0
for e in range(epochs):
    # go through all records in the training data set
    i=i+1
    print(i)
    # 遍历训练数据集中的每一条数据（即每一行）
    for record in training_data_list:
        # split the record by the ',' commas
        # 得到一个包含785个元素的列表
        all_values = record.split(',')
        # 将图像数据（除去标签的所有像素值）进行规范化处理，变换到0.01-1.00之间
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 因为sigmoid函数在接近0和1的地方梯度接近0，可能会导致学习变慢。
        targets = numpy.zeros(output_nodes) + 0.01
        # all_value[0]可以理解为这个png的tag(0,1,2,3,4,5,6,7,8,9]
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)


# 保存训练好的模型
save_object(n,"trained_n")


# load image data from png files into an array
print("loading png")
im=PIL.Image.open(r'F:\3.png')
im = im.resize((28, 28))
im.save(r'F:\3.png')
img_array = imageio.imread(r'F:\3.png', mode='F')

# reshape from 28x28 to list of 784 values, invert values
img_data = 255.0 - img_array.reshape(784)

# 之所以要进行这一步处理是因为要去除背景，使得测试数据与训练数据的像素矩阵一致。

# then scale data to range from 0.01 to 1.0
img_data = (img_data / 255.0 * 0.99) + 0.01
print("min = ", numpy.min(img_data))
print("max = ", numpy.max(img_data))

# plot image
matplotlib.pyplot.imshow(img_data.reshape(28, 28), cmap='Greys', interpolation='None')

# query the network
outputs = n.query(img_data)
print (outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network says ", label)