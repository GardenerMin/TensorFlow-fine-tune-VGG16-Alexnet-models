import numpy as np
import cv2
from math import exp, pi
import tensorflow as tf
import random
from PIL import Image

class Dataset:
    def __init__(self, train_list, test_list):
        # Load training images (path) and labels
        with open(train_list) as f:
            lines = f.readlines()
            self.train_image = []
            self.train_label = []
            for l in lines:
                items = l.split()
                self.train_image.append(items[0])
                self.train_label.append(int(items[1]))

        # Load testing images (path) and labels
        with open(test_list) as f:
            lines = f.readlines()
            self.test_image = []
            self.test_label = []
            for l in lines:
                items = l.split()
                self.test_image.append(items[0])
                self.test_label.append(int(items[1]))

        # Init params
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(self.train_label)
        self.test_size = len(self.test_label)
        self.crop_size = 224
        self.scale_size = 256
        self.mean = np.array([95.388, 112.346, 148.5382])   # alexnet: ([104., 117., 124.]) in bgr order; vggmean: [129.1863,104.7624,93.5940] in rgb order;
        self.n_classes = 10575                              #
    
    def next_batch(self, batch_size, phase):
        # Get next batch of image (path) and labels
        if phase == 'train':
            if self.train_ptr + batch_size < self.train_size:
                paths = self.train_image[self.train_ptr:self.train_ptr + batch_size]
                labels = self.train_label[self.train_ptr:self.train_ptr + batch_size]
                self.train_ptr += batch_size
            else:
                new_ptr = (self.train_ptr + batch_size)%self.train_size
                paths = self.train_image[self.train_ptr:] + self.train_image[:new_ptr]
                labels = self.train_label[self.train_ptr:] + self.train_label[:new_ptr]
                self.train_ptr = new_ptr
        elif phase == 'test':
            if self.test_ptr + batch_size < self.test_size:
                paths = self.test_image[self.test_ptr:self.test_ptr + batch_size]
                labels = self.test_label[self.test_ptr:self.test_ptr + batch_size]
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size)%self.test_size
                paths = self.test_image[self.test_ptr:] + self.test_image[:new_ptr]
                labels = self.test_label[self.test_ptr:] + self.test_label[:new_ptr]
                self.test_ptr = new_ptr
        else:
            return None, None

        # Read images
        images = np.ndarray([batch_size, self.crop_size, self.crop_size, 3])
        for i, path in enumerate(paths):
            img = cv2.imread(path)  # bgr
            h, w, c = img.shape
            if c == 3:
                img = img
            else:
                img = np.stack((img,) * 3)
            img = img.astype(np.float32)
            img -= self.mean
            if phase == 'train':
                shift_x = random.randint(0, self.scale_size - self.crop_size - 1)
                shift_y = random.randint(0, self.scale_size - self.crop_size - 1)
                images[i] = img[shift_x: shift_x + self.crop_size, shift_y: shift_y + self.crop_size]
            elif phase == 'test':
                shift = (self.scale_size - self.crop_size) // 2
                images[i] = img[shift: shift + self.crop_size, shift: shift + self.crop_size, :]

            #img = cv2.resize(img, (self.scale_size, self.scale_size))
            #shift = (self.scale_size - self.crop_size) // 2
            #img_crop = img[shift: shift + self.crop_size, shift: shift + self.crop_size, :]
            #images[i] = img  # img_crop

        # Expand labels
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i, label in enumerate(labels):
            one_hot_labels[i][label] = 1
        return images, one_hot_labels
        # Generate Lable distribution
        # one_hot_labels = np.zeros((batch_size, self.n_classes))
        # one_hot_labels_ori = np.zeros((batch_size, self.n_classes))
        # lis = np.arange(0, self.n_classes, 1)  #(18, 48, 1)  [10,...,80] SAME AS self.range, totally 71 labels   #
        # sigma = 2  # unknown
        # for i, label in enumerate(labels):  # column: self.range ; row: batch_size
        #     one_hot_labels_ori[i] = np.exp(-((lis-label)**2)/(2*sigma**2)) * (2*pi*sigma**2)**(-0.5)
        #     one_hot_labels[i] = one_hot_labels_ori[i]/sum(one_hot_labels_ori[i])

        # return images, one_hot_labels
