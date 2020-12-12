"""
 这部分介绍tf.Tensor的各种常用属性。
 张量是对矢量和矩阵向潜在的更高维度的泛化。
 TensorFlow将张量表现为基本数据类型的n维数组。
 在编写 TensorFlow 程序时，操控和传递的主要目标是 tf.Tensor。
 tf.Tensor 目标表示一个部分定义的计算，最终会产生一个值。
 TensorFlow 程序首先建立 tf.Tensor 目标图，详细说明如何基于其他可用张量来计算每个张量，
 然后运行该图的部分内容以获得所期望的结果。

 tf.Tensor 有以下属性： 数据类型和形状。
 张量中的每个元素都具有相同的数据类型，且该数据类型一定是已知的。
 形状，即张量的维数和每个维度的大小，可能只有部分已知。
 如果其输入的形状也完全已知，则大多数指令会生成形状完全已知的张量，
 但在某些情况下，只能在图的执行时间找到张量的形状。

 主要的张量包括：

 tf.Variable 变量张量，一般表示机器学习参数。
 tf.constant 常量张量，一般用来表示常量，不可改变。
 tf.placeholder 占位符张量，一般用来表示机器学习中输入的数据（训练时喂入的数据）。
"""
import tensorflow as tf
# 0阶变量，对应的shape为（）
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int64)
floating = tf.Variable(3.1415926, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)
print(mammal.shape)
# --------------------------------------------------------------------------------
# 1阶变量，对应的shape为（m，）
mystr = tf.Variable(["Hello"], tf.string)
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)
print(cool_numbers.shape) 
# --------------------------------------------------------------------------------
# 2阶变量，对应的shape为（m，n）
mymat = tf.Variable([[7],[11]], tf.int16)
myxor = tf.Variable([[False, True],[True, False]], tf.bool)
linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
rank_of_squares = tf.rank(squarish_squares)
mymatC = tf.Variable([[7],[11]], tf.int32)
print(linear_squares.shape)
# --------------------------------------------------------------------------------
"""
# reshape: 张量的元素数量是其所有形状大小的乘积，通常我们需要对张量的形状进行变换，
# 例如我们在训练模型时输入是一个1阶张量，但我们需要采用mini-batch gradient descent来进行模型训练，
# 那么就需要把1阶张量reshape成一个2阶张量，由多个batch组成。这一点可以通过tf.reshape来完成。
"""
rank_three_tensor = tf.ones([3, 4, 5])
print(rank_three_tensor.shape)
matrix = tf.reshape(rank_three_tensor, [6, 10])
print(matrix.shape)
matrix_b = tf.reshape(matrix, [3, -1])
print(matrix_b.shape)
# --------------------------------------------------------------------------------
"""
切片: n阶的tf.Tensor对内表示为n维单元数组，要访问tf.Tensor中的某一单元，则需要指定 ，与python 访问list的切片相似。
0阶张量： 本身就是标量，不需要指定，因为其本身便是单一数字。
1阶张量： 传递单一指数则可以访问一个数字。
2阶张量： 传递两个数字会如预期般返回一个标量。而传递一个数字，则会返回一个矩阵的子矢量。
"""
# --------------------------------------------------------------------------------
# 1阶变量
cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
slice_0 = cool_numbers[0]
print(slice_0)
# 请注意，如果想从矢量中动态地选择元素，那么 [] 内传递的指数本身可以是一个标量 tf.Tensor。
# 只需要能够解析即可
cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
idx = tf.Variable(1,tf.int16)
slice_0 = cool_numbers[idx]
print(slice_0.shape)
# --------------------------------------------------------------------------------
# 2阶变量
squarish_squares = tf.Variable([[4, 9], [16, 25]])