import tensorflow as tf
import numpy as np
# 定义一个计算图
g = tf.Graph()
with g.as_default():
    # 定义张量
    t1 = tf.constant(np.pi)
    t2 = tf.constant([1, 2, 3, 4])
    t3 = tf.constant([[1, 2], [3, 4]])
    # 获取张量的阶
    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)
    # 获取张量的shapes
    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()
    print("shapes:", s1, s2, s3)
# 启动前面定义的图来进行下一步操作
f = tf.Graph()
with f.as_default():
    t1 = tf.constant([1, 2, 3, 4])
    t2 = tf.constant([[1, 2], [3, 4]])

    r1 = tf.rank(t1)
    r2 = tf.rank(t2)

    s1 = t1.get_shape()
    s2 = t2.get_shape()
    print("shapes:", s1, s2)
with tf.Session(graph=g) as sess:
    print("Ranks:", r1.eval(), r2.eval())





