# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import tensorflow as tf
import sklearn.preprocessing as sp
import matplotlib.pyplot as mp
import numpy as np
import os
import cv2
import warnings


warnings.filterwarnings('ignore')


def search_data(file_path):
    '''
    get path for all images
    
    params
        file_path: directory of images
        
    returns
        faces: dict of image path, labels for keys, file path for values
    '''
    
    file_path = os.path.normpath(file_path)
    if not os.path.exists(file_path):
        raise IOError('The file path "{}" is not existed!'.format(file_path))
    faces = {}
    for dirpath, subdirs, filenames in os.walk(file_path):
        for image in (file for file in filenames if file.endswith('.jpg')):
            path = os.path.join(dirpath, image)
            label = dirpath[-1]
            if label not in faces:
                faces[label] = []
            faces[label].append(path)

    return faces


def get_data(faces):
    '''
    get gray array of images
    
    params
        faces: dict, image path 
        
    returns
        x: gray array for images
        y: labels array for images
    '''
    
    x, y = [], []
    for label, filenames in train_faces.items():
        for filename in filenames:
            image = cv2.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            x.append(gray.ravel())
            y.append([float(label) + 1.])

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    return x, y


class Model():

    '''
    create CNN model, train model, and save the trained model
    '''

    def __init__(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.x_shuffle = train_x
        self.y_shuffle = train_y
        self._index_in_epoch = 0
        self.x = tf.placeholder(tf.float32, [None, 48 * 48])
        self.y_ = tf.placeholder(tf.float32, [None, 7])
        self.keep_prob = tf.placeholder(tf.float32)

    def next_batch(self, x, y, batch_size, codec):
        standard_scaler = sp.StandardScaler()
        start = self._index_in_epoch
        num_examples = len(y)
        if self._index_in_epoch == 0 and start == 0:
            index0 = np.arange(num_examples)
            np.random.shuffle(index0)
            self.x_shuffle = x[index0]
            self.y_shuffle = y[index0]

        if start + batch_size > num_examples:
            rest_num_examples = num_examples - start
            x_rest_part = self.x_shuffle[start: num_examples]
            y_rest_part = self.y_shuffle[start: num_examples]

            index = np.arange(num_examples)
            np.random.shuffle(index)
            self.x_shuffle = x[index]
            self.y_shuffle = y[index]

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            x_new_part = self.x_shuffle[start: end]
            y_new_part = self.y_shuffle[start: end]

            x_ = standard_scaler.fit_transform(np.concatenate((x_rest_part, x_new_part), axis=0))
            y_ = codec.transform(np.concatenate((y_rest_part, y_new_part), axis=0))

            return x_, y_
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            x_ = standard_scaler.fit_transform(self.x_shuffle[start: end])
            y_ = codec.transform(self.y_shuffle[start: end])
            
            return x_, y_

    def weight_variable(self, shape):
        init = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(init)

    def bias_variable(self, shape):
        init = tf.constant(0.1, shape=shape)
        return tf.Variable(init)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pooling2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    def neuro_net(self):
        # input layer
        x_image = tf.reshape(self.x, [-1, 48, 48, 1])

        # first conv layer
        w_conv1 = self.weight_variable([3, 3, 1, 16])
        b_conv1 = self.bias_variable([16])

        out_conv1 = tf.nn.relu(self.conv2d(x_image, w_conv1) + b_conv1)
        out_pooling1 = self.max_pooling2x2(out_conv1)

        # second conv layer
        w_conv2 = self.weight_variable([3, 3, 16, 32])
        b_conv2 = self.bias_variable([32])

        out_conv2 = tf.nn.relu(self.conv2d(out_pooling1, w_conv2) + b_conv2)
        out_pooling2 = self.max_pooling2x2(out_conv2)

        # third conv layer
        w_conv3 = self.weight_variable([3, 3, 32, 64])
        b_conv3 = self.bias_variable([64])

        out_conv3 = tf.nn.relu(self.conv2d(out_pooling2, w_conv3) + b_conv3)
        out_pooling3 = self.max_pooling2x2(out_conv3)
        
        # dropout layer
        fc1_dropout = tf.nn.dropout(out_fc1, self.keep_prob)

        # first full connection layer
        w_fc1 = self.weight_variable([6 * 6 * 64, 2048])
        b_fc1 = self.bias_variable([2048])

        out_pooling3_flat = tf.reshape(out_pooling3, [-1, 6 * 6 * 64])
        out_fc1 = tf.nn.relu(tf.matmul(out_pooling3_flat, w_fc1) + b_fc1)

        # second full connection layer
        w_fc2 = self.weight_variable([2048, 1024])
        b_fc2 = self.bias_variable([1024])

        out_fc2 = tf.nn.relu(tf.matmul(fc1_dropout, w_fc2) + b_fc2)
        fc2_dropout = tf.nn.dropout(out_fc2, self.keep_prob)

        # output layer
        w_out = self.weight_variable([1024, 7])
        b_out = self.bias_variable([7])

        f_out = tf.matmul(fc2_dropout, w_out) + b_out

        # loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=f_out))
        
        # train step
        train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

        # accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_, 1), tf.argmax(f_out, 1)), tf.float32))

        return train_step, accuracy, cross_entropy

    def train_model(self, train_step, accuracy, cross_entropy):
        codec = sp.OneHotEncoder(sparse=False, dtype=float)
        codec.fit(self.train_y)
        accs, losses = [], []
        
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            for i in range(10000):
                batch_x, batch_y = self.next_batch(self.train_x, self.train_y, 32, codec=codec)
                if i % 100 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob: 1.0})
                    print('step {}, the train accuracy: {}'.format(i, train_accuracy))
                    
                    checkpoint_path = 'models'
                    if os.path,exists(checkpoint_path):
                        os.makedirs(checkpoint_path)
                    saver.save(sess, os.path.join(checkpoint, 'model.ckpt'), global_step=i)
                    
                acc = sess.run(accuracy, feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob: 1.0})
                loss = sess.run(cross_entropy, feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob: 1.0})
                accs.append(acc)
                losses.append(loss)

                sess.run(train_step, feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob: 0.5})
                
            batch_x, batch_y = self.next_batch(self.test_x, self.test_y, len(self.test_y), codec=codec)
            test_accuracy = sess.run(accuracy, feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_prob: 1.0})
            print('the test accuracy: {}'.format(test_accuracy))

            return accs, losses


def main():
    train_faces = search_data('datasets/train')
    train_x, train_y = get_data(train_faces)
    test_faces = search_data('datasets/test')
    test_x, test_y = get_data(test_faces)

    model = Model(train_x, train_y, test_x, test_y)
    train_step, accuracy, cross_entropy = model.neuro_net()
    accuracy, loss = model.train_model(train_step, accuracy, cross_entropy)

    x1 = np.arange(len(accuracy))
    x2 = np.arange(len(loss))
    mp.figure('accuracy')
    mp.title('accuracy', fontsize=20)
    mp.xlabel('batch', fontsize=14)
    mp.ylabel('accuracy', fontsize=14)
    mp.plot(x1, accuracy, c='orangered')

    mp.figure('loss')
    mp.title('loss', fontsize=20)
    mp.xlabel('batch', fontsize=14)
    mp.ylabel('loss', fontsize=14)
    mp.plot(x2, loss, c='blue')

    mp.show()


if __name__ == '__main__':
    main()
