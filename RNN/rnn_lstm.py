
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

#设置GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


# In[3]:


#导入数据
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
print(mnist.train.images.shape)


# In[5]:


#设置好模型用到的各个超参数
lr=1e-3
#在训练和测试的时候，我们想用不同的 batch_size.所以采用占位符的方式
batch_size = tf.placeholder(tf.int32,[])
keep_prob = tf.placeholder(tf.float32,[])

# 每个时刻的输入特征是28维的，就是每个时刻输入一行，一行有 28 个像素
input_size = 28
# 时序持续长度为28，即每做一次预测，需要先输入28行
timestep_size=28
#每个隐含层的节点数
hidden_size=256
# LSTM layer 层数
layer_num = 2
#最后输出分类类别数量，如果是回归预测的话应该是 1
class_num = 10
with tf.Graph().as_default(): 
    _X = tf.placeholder(tf.float32,[None,784])
    y = tf.placeholder(tf.float32,[None,class_num])
    keep_prob = tf.placeholder(tf.float32)
    #开始搭建 LSTM 模型，其实普通 RNNs 模型也一样
    # 把784个点的字符信息还原成 28 * 28 的图片
    # 下面几个步骤是实现 RNN / LSTM 的关键
    # **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size) 
    X = tf.reshape(_X,[-1,28,28])
    # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=1.0,state_is_tuple=True)
    # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell,input_keep_prob=1.0,output_keep_prob=keep_prob)
    # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    mlstm_cell = rnn.MultiRNNCell([lstm_cell]*layer_num,state_is_tuple=True)
    #**步骤5：用全零来初始化state
    init_state = mlstm_cell.zero_state(batch_size,dtype=tf.float32)
    #步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
    outputs,state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
    h_state = outputs[:,-1,:]  # 或者 h_state = state[-1][1]
# print(h_state.shape)
    #设置 loss function 和 优化器，展开训练并完成测试
    #我们要分类的话，还需要接一个 softmax 层
    # 首先定义 softmax 的连接权重矩阵和偏置
    W = tf.Variable(tf.truncated_normal([hidden_size,class_num],stddev=0.1),dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1,shape=[class_num]),dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h_state,W)+bias)

    # 损失 评估
    cross_entropy = -tf.reduce_mean(y*tf.log(y_pre))
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y,1))
    accuracy = tf.reduct_mean(tf.cast(correct_prediction,"float32"))

    sess.run(tf.gloal_variables_initializer())
    for i in range(2000):
        _batch_size = 128
        batch = mnist.train.next_batch(_batch_size)
        print(batch[0])
        if (i+1)%200 == 0:
            train_accuracy = sess.run(accuracy,feed_dict={_X:batch[0],y:batch[1],keep_prob:1.0,batch_size:_batch_size})
            print("Iter%d, step %d, training accuracy %g" % ( mnist.train.epochs_completed, (i+1), train_accuracy))

    print(mnist.test.images.shape[0])
    # test
    print(" test accuracy %g" % (  sess.run(accuracy,feed_dict={
        _X:mnist.test.image,y:mnist.test.labels,keep_prob:1.0,batch_size:mnist.test.images.shape[0]})))


