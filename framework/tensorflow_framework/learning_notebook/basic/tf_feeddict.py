import tensorflow as tf


# 定义占位符,定义好占位符之后在后面的sess run 中需要喂入数据，否则报错。
x = tf.placeholder(shape=(1, 2), dtype=tf.float32)

# 这里本来应该声明的是两个variable变量，不小心写成了常量。不过可以正好做个测试。
w1 = tf.random_normal(shape=[2, 3], mean=0.0, stddev=1.0, seed=0)
w2 = tf.random_normal(shape=[3, 1], mean=0.0, stddev=1.0, seed=0)

# 声明两个变量，这两个变量是权重变量，在计算过程中会时时更新的，所以用varibale表示。
w3 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0.0, stddev=1.0, seed=1))
w4 = tf.Variable(tf.random_normal(shape=[3, 1], mean=0.0, stddev=1.0, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    # 当计算图中有placeholder占位符时，在进行sess run时必须通过feed dict 给其喂入数据，否则提示错误。
    # 但是在这个时候可以不执行：tf.global_variables_initializer()函数，因为没有variable变量。
    print(sess.run(y, feed_dict={x:[[2.0, 3.0]]}))

b = tf.matmul(x, w3)
z = tf.matmul(b, w4)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 直接run 什么都没有的时候提示错误,FailedPreconditionError: Attempting to use uninitialized value Variable。
    print(sess.run(z, feed_dict={x: [[1.0, 3.0]]}))
