import tensorflow as tf
import numpy as np
import sys
from network import *
import tensorflow.contrib.slim as slim

class Model:
    @staticmethod 
    def vgg16(net_input, keep_rate, n_classes):
        # TODO weight decay loss tern
        # 1st Layer: Conv -> Conv -> Pool
        conv1_1 = conv(net_input, 3, 3, 64, 1, 1, padding='SAME', name='conv1_1', group=1, trainable=True)
        conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, padding='SAME', name='conv1_2', group=1, trainable=True)
        pool1 = max_pool(conv1_2, 2, 2, 2, 2, padding='SAME', name='pool1')

        # 2nd Layer: Conv -> Conv -> Pool
        conv2_1 = conv(pool1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_1', group=1, trainable=True)
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, padding='SAME', name='conv2_2', group=1, trainable=True)
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, padding='SAME', name='pool2')

        # 3rd Layer: Conv -> Conv -> Conv -> Pool
        conv3_1 = conv(pool2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_1', group=1, trainable=True)
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, padding='SAME', name='conv3_2', group=1, trainable=True)
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, padding='SAME', name='conv3_3', group=1, trainable=True)
        pool3 = max_pool(conv3_3, 2, 2, 2, 2, padding='SAME', name='pool3')


        # 4th Layer: Conv -> Conv -> Conv -> Pool
        conv4_1 = conv(pool3, 3, 3, 512, 1, 1, padding='SAME', name='conv4_1', group=1, trainable=True)
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, padding='SAME', name='conv4_2', group=1, trainable=True)
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, padding='SAME', name='conv4_3', group=1, trainable=True)
        pool4 = max_pool(conv4_3, 2, 2, 2, 2, padding='SAME', name='pool4')


        # 5th Layer: Conv -> Conv -> Conv -> Pool
        conv5_1 = conv(pool4, 3, 3, 512, 1, 1, padding='SAME', name='conv5_1', group=1, trainable=True)
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, padding='SAME', name='conv5_2', group=1, trainable=True)
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, padding='SAME', name='conv5_3', group=1, trainable=True)
        pool5 = max_pool(conv5_3, 2, 2, 2, 2, padding='SAME', name='pool5')



        # 6th Layer: FC -> DropOut
        # [1:] cuts away the first element
        #pool5_out = int(np.prod(pool5.get_shape()[1:])) # 7 * 7 * 512 = 25088
        #pool5_flat = tf.reshape(pool5, [-1, pool5_out]) # shape=(image count, 7, 7, 512) -> shape=(image count, 25088)
        #fc6 = fc(pool5_flat, 4096, name='fc6')
        fc6 = fc1(pool5, 4096, name='fc6', trainable=True)
        dropout1 = tf.nn.dropout(fc6, keep_rate)

        # 7th Layer: FC
        fc7 = fc(dropout1, 4096, 4096, name='fc7', trainable=True)
        dropout2 = tf.nn.dropout(fc7, keep_rate)

        # 8th Layer: FC
        fc8 = fc(dropout2, 4096, n_classes, name='fc8', relu=False, trainable=True)

        return fc8

    def alexnet(net_input, keep_rate, n_classes):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool-lrn)
        conv1 = conv(net_input, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
        # Layer 2 (conv-relu-pool-lrn)
        conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
        conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
        # Layer 3 (conv-relu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        # Layer 4 (conv-relu)
        conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
        # Layer 5 (conv-relu-pool)
        conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        # Layer 6 (fc-relu-drop)
        fc6 = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
        fc6 = dropout(fc6, keep_rate)
        # Layer 7 (fc-relu-drop)
        fc7 = fc(fc6, 4096, 4096, name='fc7')
        fc7 = dropout(fc7, keep_rate)
        # Layer 8 (fc-prob)
        fc8 = fc(fc7, 4096, n_classes, relu=False, name='fc8')
        return fc8
