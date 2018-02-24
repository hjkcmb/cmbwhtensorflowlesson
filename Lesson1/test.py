import tensorflow as tf

x=tf.constant([[1,2]])
y=tf.constant([[3],[4]])
s=tf.matmul(x,y)

sess=tf.Session()
print(x)
print(sess.run(s))