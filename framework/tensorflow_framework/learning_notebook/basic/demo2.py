import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

g = tf.Graph()
with g.as_default():
    v_1 = tf.constant([1, 2, 3, 4, 5])
    v_2 = tf.constant([3, 5, 2, 4, 1])
    v_add = tf.add(v_1, v_2)
    weight = tf.Variable(tf.random_normal((100, 100), stddev=2))
    weight2 = tf.Variable(weight.initialized_value(), name="w2")
with tf.Session(graph=g) as sess:
    print(sess.run(v_add))
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(weight))
    print(sess.run(weight2))
