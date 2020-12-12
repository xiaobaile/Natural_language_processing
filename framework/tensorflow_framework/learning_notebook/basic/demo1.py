import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

message = tf.constant("Welcome to the exciting world of Deep Neural Networks!")

with tf.Session() as sess:
    print(sess.run(message).decode())
