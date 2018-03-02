#! /usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import time
from tensorflow.python.ops import variables

filename_train = 'train.tfrecords'
filename_test = 'test.tfrecords'
train_dir = 'pre'
test_dir = '/资料/pre_test'


def read_conf(dict):
    with open(dict, 'r', encoding='utf-8') as file1:
        d = file1.read().splitlines()
    return d


def read_and_decode(filename):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label


batch_size = 128
model_file_name = "model/model_2018_2_28.ckpt"
# load model, resume from previous checkpoint
resume = True
with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    x_train_, y_train_ = read_and_decode(filename_train)
    x_test_, y_test_ = read_and_decode(filename_test)
    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_, y_train_],
                                                          batch_size=batch_size, capacity=2000, min_after_dequeue=1000,
                                                          num_threads=4)  # set the number of threads here
    print(x_train_batch, y_train_batch)
    x_test_batch, y_test_batch = tf.train.batch([x_train_, y_train_],
                                                batch_size=batch_size, capacity=2000,
                                                num_threads=4)  # set the number of threads here


    def inference_batch_norm(x_crop, y_, reuse, is_train):
        # 构建CNN训练模型
        W_init = tf.contrib.layers.xavier_initializer()
        W_init2 = tf.truncated_normal_initializer(stddev=0.04)
        b_init2 = tf.constant_initializer(value=0.1)
        with tf.variable_scope("model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net = tl.layers.InputLayer(x_crop, name='input')
            net = tl.layers.Conv2dLayer(net, act=tf.identity, shape=[3, 3, 3, 32],
                                        strides=[1, 1, 1, 1], padding='SAME',
                                        W_init=W_init, name='cnn_layer1')
            net = tl.layers.BatchNormLayer(net, is_train=is_train, name='batch_norm1')
            net.outputs = tf.nn.relu(net.outputs, name='relu1')
            net = tl.layers.PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME', pool=tf.nn.max_pool, name='pool1')

            net = tl.layers.Conv2dLayer(net, act=tf.identity, shape=[3, 3, 32, 64],
                                        strides=[1, 1, 1, 1], padding='SAME',
                                        W_init=W_init, name='cnn_layer2')
            net = tl.layers.BatchNormLayer(net, is_train=is_train, name='batch_norm2')
            net.outputs = tf.nn.relu(net.outputs, name='relu2')
            net = tl.layers.PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME', pool=tf.nn.max_pool, name='pool2')

            net = tl.layers.Conv2dLayer(net, act=tf.identity, shape=[3, 3, 64, 64],
                                        strides=[1, 1, 1, 1], padding='SAME',
                                        W_init=W_init, name='cnn_layer3')
            net = tl.layers.BatchNormLayer(net, is_train=is_train, name='batch_norm3')
            net.outputs = tf.nn.relu(net.outputs, name='relu3')
            net = tl.layers.PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                      padding='SAME', pool=tf.nn.max_pool, name='pool3')

            net = tl.layers.FlattenLayer(net, name='flatten_layer')

            net = tl.layers.DenseLayer(net, n_units=1024, act=tf.nn.relu,
                                       W_init=W_init2, b_init=b_init2, name='d1relu')
            net = tl.layers.DenseLayer(net, n_units=192,
                                       act=tf.nn.relu, W_init=W_init2, b_init=b_init2, name='d2relu')
            if is_train:
                # 训练过程防止过拟合
                net = tl.layers.DropoutLayer(net, keep=0.5, is_fix = True,name='drop1')
                net = tl.layers.DenseLayer(net, n_units=7,
                                           act=tl.activation.identity,
                                           W_init=tf.truncated_normal_initializer(stddev=1 / 192.0),
                                           name='output')
            else:
                net = tl.layers.DenseLayer(net, n_units=7,
                                           act=tf.identity, W_init=tf.truncated_normal_initializer(stddev=1 / 192.0),
                                           name='output')

            y = net.outputs
            # 损失函数
            cost = tl.cost.cross_entropy(y, y_, name='softmaxloss')
            L2 = tf.contrib.layers.l2_regularizer(0.004)(net.all_params[6]) + \
                 tf.contrib.layers.l2_regularizer(0.004)(net.all_params[8])
            cost = cost + L2
            correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return cost, acc, net


    with tf.device('/gpu:0'):
        #调用BN后的模型
        cost, acc, network = inference_batch_norm(x_train_batch, y_train_batch, None, is_train=True)
        cost_test, acc_test, _ = inference_batch_norm(x_test_batch, y_test_batch, True, is_train=False)

    n_epoch = 50
    learning_rate = 0.001
    print_freq = 1
    batch_size = 128
    train = read_conf('dict.txt')
    test = read_conf('dict2.txt')
    n_step_epoch = int(int((train[0].split(':'))[1]) / batch_size)
    n_step = n_epoch * n_step_epoch

    with tf.device('/gpu:0'):  # 使用GPU
        # 在GPU上训练
        train_params = network.all_params  # 待训练的参数为所有的网络参数
        # 定义训练操作，使用自适应矩估计（ADAM）算法最小化损失函数
        train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                          epsilon=1e-08, use_locking=False).minimize(cost)

    tl.layers.initialize_global_variables(sess)
    print(variables._all_saveable_objects())
    if resume:
        print("Load existing model " + "!" * 10)
        saver = tf.train.Saver()
        saver.restore(sess, model_file_name)

    network.print_params(False)
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

    coord = tf.train.Coordinator()  # 创建一个线程协调器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 创建线程
    # for step in range(n_step):
    step = 0
    for epoch in range(n_epoch):  # 对每一个epoch
        start_time = time.time()  # 计时开始
        train_loss, train_acc, n_batch = 0, 0, 0  # 初始化为0
        for s in range(n_step_epoch):  # 对每一次迭代
            ## You can also use placeholder to feed_dict in data after using
            err, ac, _ = sess.run([cost, acc, train_op])
            step += 1
            train_loss += err
            train_acc += ac
            n_batch += 1
        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:  # 按预定的打印频率打印训练信息
            print("Epoch %d : Step %d-%d of %d took %fs" % (
                epoch, step, step + n_step_epoch, n_step, time.time() - start_time))
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))

            test_loss, test_acc, n_batch = 0, 0, 0  # 打印测试信息
            for _ in range(int(int((test[0].split(':'))[1]) / batch_size)):
                err, ac = sess.run([cost_test, acc_test])
                test_loss += err
                test_acc += ac
                n_batch += 1
            print("   test loss: %f" % (test_loss / n_batch))
            print("   test acc: %f" % (test_acc / n_batch))

        if (epoch + 1) % (print_freq * 10) == 0:  # 定义保存model的时机
            print("Save model " + "!" * 10)
            saver = tf.train.Saver()
            save_path = saver.save(sess, model_file_name)
    coord.request_stop()  # 请求终止所有线程
    coord.join(threads)  # 等待终止所有线程
    sess.close()  # 关闭会话
