import tensorflow as tf

x=tf.constant([[1,2]])
y=tf.constant([[3],[4]])
s=tf.matmul(x,y)

# sess=tf.Session()
# print(sess.run(s))
# sess.close()

# with tf.Session() as sess:
#     print(sess.run(s))