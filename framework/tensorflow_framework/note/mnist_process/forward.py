import tensorflow as tf

"""
前向传播过程中，需要定义神经网络中的参数 w 和偏置 b，定义由输入到输出的网络结构。
定义网络模型输入层个数、隐藏层节点数、输出层个数。
通过定义函数 get_weight()实现对参数 w 的设置，包括参数 w 的形状和是否正则化的标志。
同样，通过定义函数 get_bias()实现对偏置 b 的设置。
"""

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2
    # 由于输出 y 要经过 softmax 函数，使其符合概率分布，故输出 y 不经过 relu 函数。
    return y


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b
































