import tensorflow as tf


# tf.constant 用来定义常量，一共涉及到的变量有常量，变量和占位符。
# 常量是在计算过程中其值是不会变化的。
# 变量是在计算过程中其值是变化的，在深度学习模型中，通常权重和偏置用变量表示。
# 占位符用来存储数据的，在没有会话时，只是起到占位作用，当启动会话时可以通过feed_dict={}的形式喂入数据。

a = tf.constant([2.0, 3.0])
b = tf.constant([1.0, 2.0])

# 实现a与b的加法。
c = a + b
# 没有创建会话，不会执行计算。
print(c)
# 创建会话才会执行计算。
with tf.Session() as sess:
    print(sess.run(c))
