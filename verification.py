import tensorflow as tf
import tensorlayer as tl
from scipy.misc import imread, imresize
import numpy as np


model_file_name = "/billtrain/model/model_cifar10_advanced.ckpt"

def read_conf(dict):
    with open(dict, 'r', encoding='utf-8') as file1:
        d = file1.read().splitlines()

    dicts = {}
    for i in range(1,len(d)):
        str = d[i].split(':')
        dicts[str[0]] = str[1]
    return dicts

def print_prob(prob):
    synset = read_conf('dict.txt')
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = synset[str(pred[0])]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    return top1


with tf.device('/gpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))


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
                net = tl.layers.DropoutLayer(net, keep=0.5, is_fix=True, name='drop1')
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


    x_crop = tf.placeholder(tf.float32, shape=[1, 32, 32, 3])
    y_ = tf.placeholder(tf.int32, shape=[1, ])
    cost, acc, network = inference_batch_norm(x_crop, y_, None, is_train=False)
    tl.files.load_ckpt(sess=sess, mode_name='model_cifar10_advanced.ckpt', var_list=network.all_params,
                       save_dir='/billtrain/model/',
                       is_latest=False, printable=True)
    probs = tf.nn.softmax(network.outputs)
    image = imread('/资料/pre/2211&20&7f16&8f91/2211&20&7f16&8f91_0_22.png', mode='RGB')
    img1 = imresize(image, (32, 32))
    # gag-verificationcode-deepstudy = convert2gray(gag-verificationcode-deepstudy)
    # captcha_image = gag-verificationcode-deepstudy.flatten() / 255
    prob = sess.run(probs,feed_dict={x_crop: [img1]})
    print_prob(prob[0])

