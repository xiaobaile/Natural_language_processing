import tensorflow as tf

g = tf.Graph()
with g.as_default():
    a = tf.constant([[1.0, 2.0]])
    w = tf.constant([[1.0], [2.0]])

    y = tf.matmul(a, w)
    print(y)

a = tf.constant([[2.0, 2.0]])
w = tf.constant([[1.0], [2.0]])

z = tf.matmul(a, w)
print(z)


with tf.Session() as sess:
    print(sess.run(z))