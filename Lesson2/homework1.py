import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt

# 固定随机种子
tf.set_random_seed(1)
np.random.seed(1)

# 训练数据
x = np.linspace(-1, 1, 20)[:, np.newaxis]
y = x + 0.3*np.random.randn(20)[:, np.newaxis]

# 测试数据
test_x = x.copy()
test_y = test_x + 0.3*np.random.randn(20)[:, np.newaxis]

# 显示图表
plt.scatter(x, y, c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# 传入数据


# 神经网络，2层+Dropout


#定义loss和optimizer


#初始化参数


#训练

