from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import  matplotlib.pyplot as plt

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
#增加隐藏层和输出层

#自己写

