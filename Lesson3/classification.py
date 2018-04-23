# MNIST分类问题,
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
#显示图
plt.subplot(221)
plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[2]))
plt.legend(loc='best')

plt.subplot(222)
plt.imshow(mnist.train.images[1].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[1]))
plt.legend(loc='best')

plt.subplot(223)
plt.imshow(mnist.train.images[2].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[2]))
plt.legend(loc='best')

plt.subplot(224)
plt.imshow(mnist.train.images[3].reshape((28, 28)), cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[3]))
plt.legend(loc='best')

plt.show()

# 定义softmax回归模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))
y = tf.matmul(x, W) + b


y_ = tf.placeholder(tf.float32, [None, 10])
# 定义交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
# 采用SGD优化器
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 定义会话
sess=tf.Session()
sess.run(tf.global_variables_initializer())
# 训练1000次,每次随机抓取100个
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# 评估训练好的模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 计算模型准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
