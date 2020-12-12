import tensorflow as tf


# 这里定义了占位符，其中占位符的shape有一个维度为空，和前面的不同，所以可以一次喂入多组数据。
# 当然也可以只喂入一组数据，需要使用feed dict 参数。
x = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# 这里仍然定义了变量，所以需要有变量初始化的事情要考虑。
w1 = tf.Variable(tf.random_normal([2, 3], seed=0, stddev=1.0))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1.0, seed=1))

# 进行矩阵的点积运算。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 通过会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # print(sess.run(y, feed_dict={x: [[1.0, 2.0]]}))
    print(sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4]]}))
    # 在进行w1的运算时不需要喂入x的数据，因为是一个变量，没有涉及到占位符的问题
    print("w1 result:\n", sess.run(w1))
    # 在进行a的运算时需要喂入数据，因为a的计算图中涉及到x，如果不喂入数据，会提示错误。
    print("a result:\n", sess.run(a, feed_dict={x: [[0.3, 1.2]]}))