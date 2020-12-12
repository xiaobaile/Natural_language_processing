import tensorflow as tf 
import numpy as np


# 定义每次喂入模型的数据样本的数量以及随机数生成的种子，确保产生的随机数在每次运行时都是相同的。
BATCH_SIZE = 8
SEED = 23455

# 通过确定随机种子的方法确定随机数。
rdm = np.random.RandomState(SEED)
# 定义输入变量的大小是32个样本数量，2个特征数量。
X = rdm.rand(32, 2)
# 根据构造的特征条件进行标签的确定。
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print("X:\n", X)
print("Y_:\n", Y_)

# 图计算过程中，喂入的数据用占位符表示，有占位符的计算图，在计算的时候需要喂入数据。
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

# 定义了权重，在计算图中权重和偏置用变量来声明，在sess run 中需要通过全局变量初始化开始计算。
w1 = tf.Variable(tf.random_normal(shape=(2, 3), stddev=1, seed=0))
w2 = tf.Variable(tf.random_normal(shape=(3, 1), stddev=1, seed=0))

# 定义矩阵乘法。
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义了损失函数，是均方差函数。
loss_mse = tf.reduce_mean(tf.square(y - y_))
# 定义优化器，对参数进行反向传递。
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

# 通过会话开始计算。
with tf.Session() as sess:
    # 由于前面的权重和偏置是通过变量来定义的，因此这里必须有这条全局变量初始化，方可计算。
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")

    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y_[start: end]})
        if i % 500 == 0:
            total_loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d training step(s), loss_mse on all data is %g" % (i, total_loss))

    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    # ，只搭建承载计算过程的
    # 计算图，并没有运算，如果我们想得到运算结果就要用到“会话 Session()”了。
    # √会话（Session）： 执行计算图中的节点运算
    print("w1:\n", w1)
    print("w2:\n", w2)




