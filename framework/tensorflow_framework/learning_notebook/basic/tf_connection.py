import tensorflow as tf


# 定义x是一个常量
x = tf.constant([[2.0, 3.0]])

# 定义两个权重，在tensor flow 中需要在后面计算过程中更新的变量。
# 同时注意，在计算中如果涉及到变量，必须在跑整个计算之前进行全局变量的初始化。
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w = tf.Variable(tf.random_uniform(shape=[2, 3], dtype=tf.float32))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, mean=0.0, seed=0))

# 计算点积。在张量计算时要算好维度的问题。x--->1@2, w1--->2@3, a--->1@3, w2--->3@1, y--->1@1
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 这里只是打印出结果，并没有开始计算。
print(y)

# 用会话计算结果。
with tf.Session() as sess:
    # 在计算图中存在变量的时候，必须使用这句将变量进行初始化，然后sess run 否则会报错。
    init_op = tf.global_variables_initializer()
    # 先执行变量初始化计算。
    sess.run(init_op)
    # 打印计算结果。
    print(sess.run(y))