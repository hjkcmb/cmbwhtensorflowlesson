# MNIST分类问题,
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
BATCH_SIZE = 100
LR = 0.001
drop_hide = 0.5

x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.float32, [None, 10])
tf_is_training = tf.placeholder(tf.bool, None)
# 输入shape【28，28，1】
image = tf.reshape(x, [-1, 28, 28, 1])

# ALEXNET算法

# 第一个卷积层
# 输入的图片大小为:28*28*1
# 第一个卷积层为:11*11*64,有64个卷积核,步长为1,same填充，卷积层后跟ReLU
# 最大池化层,核大小为2*2,步长为2,same填充

# 第二层卷积层
# 输入的tensor为14*14*64
# 卷积和的大小为: 5*5*192,步长为1,same填,卷积层后跟ReLU
# 最大池化层,核大小为2*2,步长为2,same填充

# 第三层至第五层卷积层
# 输入的tensor为7*7*256
# 第三层卷积为 3*3*384,步长为1,same填充,加上ReLU
# 第四层卷积为 3*3*384,步长为1,same填充,加上ReLU
# 第五层卷积为 3*3*256,步长为1,same填充,加上ReLU
# 第五层后跟最大池化层,核大小2*2,步长为2

# 第六层至第八层全连接层
# 输入的tensor为4*4*256
# 第六层全连接层为 4096 + ReLU + DropOut
# 第七层全连接层为 4096 + ReLU + DropOut
# 第八层全连接层，输出结果10

# softmax分类和交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y))
train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

# 用平均值来统计测试准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 开始训练
for step in range(1000):
    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_op, {x: batch_x, y: batch_y})
    if step % 10 == 0:
        accuracy_ = sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels})
        print('Step:', step, '| test accuracy: ', accuracy_)
