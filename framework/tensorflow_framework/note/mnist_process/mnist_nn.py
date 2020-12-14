from tensorflow.examples.tutorials.mnist import input_data

""" 有用的函数
tf.get_collection("")：表示从collection集合中取出全部变量组成一个列表
tf.add()：表示将参数列表中对应元素相加
tf.cast(x, dtype)：表示将参数x转换为指定数据类型
tf.equal()：表示对比两个矩阵或者向量元素。若对应元素相等，则返回True，若对应元素不相等，则返回False
tf.reduce_mean(x, axis)：表示求取矩阵或者张量指定维度的平均值。若不指定第二个参数，则在所有元素中取平均值；
                        若指定第二个参数为0，则在第一维元素上取平均值，即每一列求平均值
tf.argmax(x, axis)：表示返回指定维度axis下，参数x中最大值索引
os.path.join()：表示把参数字符串按照路径命名规则拼接
tf.Graph().as_default()：表示将当前图设置成默认图，并返回一个上下文管理器。一般与with关键字搭配使用，用于将已经定义好的神经网络在计算图中复线

"""

"""
神经网络模型的保存
    在反向传播过程中，一般会间隔一定轮数保存一次神经网络模型，并产生三个文件
    (保存当前图结构的.meta 文件、保存当前参数名的.index 文件、保存当 前参数的.data 文件)，在 Tensorflow 中如下表示:
        saver = tf.train.Saver() 
        with tf.Session() as sess: 
            for i in range(STEPS):
                if i % 轮数 == 0:
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    其中，tf.train.Saver()用来实例化 saver 对象。上述代码表示，神经网络每循环规定的轮数，将神经网络模型中所有的参数等信息保存到指定的路径中，
    并在存放网络模型的文件夹名称中注明保存模型时的训练轮数。
神经网络模型的加载 
    在测试网络效果时，需要将训练好的神经网络模型加载，在 Tensorflow 中这样表示:
        with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(存储路径) 
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
    在 with 结构中进行加载保存的神经网络模型，若 ckpt 和保存的模型在指定路径中存在，则将保存的神经网络模型加载到当前会话中。

加载模型中参数的滑动平均值 
    在保存模型时，若模型中采用滑动平均，则参数的滑动平均值会保存在相应文件中。
    通过实例化 saver 对象，实现参数滑动平均值的加载，在 Tensorflow 中如下表示:
        ema = tf.train.ExponentialMovingAverage(滑动平均基数)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

神经网络模型准确率评估方法 
    在网络评估时，一般通过计算在一组数据上的识别准确率，评估神经网络的效果。在 Tensorflow 中这样表示:
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    在上述中，y 表示在一组数据(即 batch_size 个数据)上神经网络模型的预测结果，y 的形状为[batch_size,10]，每一行表示一张图片的识别结果。
    通过 tf.argmax()函数取出每张图片对应向量中最大值元素对应的索引值，组成长度为输入数据 batch_size 个的一维数组。
    通过 tf.equal()函数判断预测结果张量 和实际标签张量的每个维度是否相等，若相等则返回 True，不相等则返回 False。 
    通过 tf.cast()函数将得到的布尔型数值转化为实数型，再通过 tf.reduce_mean()函数求平均值，最终得到神经网络模型在本组数据上的准确率。
"""


mnist = input_data.read_data_sets("./data/", one_hot=True)
BATCH_SIZE = 200
xs, ys = mnist.train.next_batch(BATCH_SIZE)






























