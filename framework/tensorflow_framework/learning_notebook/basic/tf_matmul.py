import tensorflow as tf


w = tf.constant([[1.0, 2.0]])
x = tf.constant([[1.0], [2.0]])
# 构建计算图。
y = tf.matmul(w, x)

print(y)

with tf.Session() as sess:
    # 执行计算图中的节点运算。
    print(sess.run(y))

