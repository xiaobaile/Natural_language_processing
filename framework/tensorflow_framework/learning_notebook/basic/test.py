import tensorflow as tf
import numpy as np

SEED=1
BATCH_SIZE=8
COST = 9
PROFIT = 1


rnd = np.random.RandomState(seed=SEED)
X = rnd.randn(32, 2)
Y = [[x0 + x1 + rnd.rand()/10 - 0.05] for (x0, x1) in X]
# print(X)
# print(Y)

x = tf.placeholder(tf.float32, shape=[None, 2], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, 1], name="y_")

w1 = tf.Variable(tf.random_normal((2,1), mean=0.0, stddev=1.0, dtype=tf.float32, name="w1", seed=0))
# w2 = tf.Variable(tf.random_normal((3,1), mean=0.0, stddev=1.0, dtype=tf.float32, name="w2", seed=0))

y = tf.matmul(x, w1)
# y = tf.matmul(a, w2)

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), COST*(y-y_), PROFIT*(y_ - y)))
step = tf.train.GradientDescentOptimizer(learning_rate=.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("before training w1 is:\n", sess.run(w1))
    # print("before training w2 is:\n", sess.run(w2))

    EPOCHES = 30000
    for i in range(EPOCHES):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            print("after %d epoch, loss_mse is %g:\t" % (i, sess.run(loss, feed_dict={x: X, y_: Y})))
            # print("After %d epoch, w1 is :\t" % i)
            # print(sess.run(w1))
    print("after 30000 epoches, the value of w1 is :\n", sess.run(w1))
    # print("after 30000 epoches, the value of w2 is :\n", sess.run(w2))
