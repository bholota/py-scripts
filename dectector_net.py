# -*- encoding: utf-8 -*-
# !/bin/python3

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib.image import imread
from six.moves import cPickle as pickle


class DetectorNet:

    def __init__(self, data_path_list, label_path_list, img_size=256, cache=False):
        self.data_dir = data_path_list
        self.img_size = img_size
        self.X, self.Y = None, None

        if cache is True:
            self.X, self.Y = self.load_from_cache()

        if self.X is None or self.Y is None:
            self.X = self.load_resized_data(data_path_list)
            self.Y = self.load_resized_data(label_path_list, channel_count=1, reshape=True)
            self.write_cache()

        key, img = next(iter(self.X.items()))
        img = img * 255
        lab = self.Y[key].reshape(256, 256)
        fig, ax = plt.subplots(1, 2, sharey=True, sharex=True)
        ax1, ax2 = ax.ravel()
        ax1.imshow(img)
        ax2.imshow(lab, cmap="Greys")
        ax1.axis('off')
        ax2.axis('off')
        plt.show()

    def train(self):
        pass

    def _loss(self):
        pass

    def get_augmentation(self, data):
        pass

    def load_from_cache(self):
        if os.path.exists("X.pickle") and os.path.exists("Y.pickle"):
            try:
                X, Y = None, None
                with open("X.pickle", "rb") as f:
                    X = pickle.load(f)
                with open("Y.pickle", "rb") as f:
                    Y = pickle.load(f)
                return X, Y
            except Exception as e:
                print("* Loading from cache failed")
                return None, None
        else:
            return None, None

    def write_cache(self):
        try:
            with open('X.pickle', "wb") as f:
                pickle.dump(self.X, f, pickle.HIGHEST_PROTOCOL)
            with open('Y.pickle', "wb") as f:
                pickle.dump(self.Y, f, pickle.HIGHEST_PROTOCOL)
            return True
        except Exception as e:
            print("* Writing cache failed")
            return False

    def load_resized_data(self, dir_path, channel_count=3, reshape=False):
        x_data = {}
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, (None, None, channel_count))
        normalized = tf.image.per_image_standardization(X)
        tf_img = tf.image.resize_images(X, (self.img_size, self.img_size),
                                        tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for index, file_path in enumerate(dir_path):
                img = imread(file_path)
                img = img.reshape([img.shape[0], img.shape[1], channel_count]) if reshape else img
                resized_img = sess.run(tf_img, feed_dict={X: img[:, :, :channel_count]})
                key = file_path.split('/')[-1].split('.')[-2].replace('_heatmap', '')
                x_data[key] = resized_img

        return x_data
