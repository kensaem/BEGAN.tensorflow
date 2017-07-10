import os
import glob
import collections
import cv2
import numpy as np
import tensorflow as tf

BatchTuple = collections.namedtuple("BatchTuple", ['images', 'labels'])


class Loader:
    RawDataTuple = collections.namedtuple("RawDataTuple", ['path', 'label'])

    def __init__(self, data_path, batch_size):
        self.sess = tf.Session()
        self.image_info = {
            'width': 32,
            'height': 32,
            'channel': 3,
        }

        self.data = []
        self.data_path = data_path
        self.batch_size = batch_size
        self.cur_idx = 0
        self.perm_idx = []
        self.epoch_counter = 0

        self.load_data()

        self.reset()
        return

    def load_data(self):
        # Load data from directory
        print("...Loading from %s" % self.data_path)
        dir_name_list = os.listdir(self.data_path)
        for dir_name in dir_name_list:
            dir_path = os.path.join(self.data_path, dir_name)
            file_name_list = os.listdir(dir_path)
            print("\tNumber of files in %s = %d" % (dir_name, len(file_name_list)))
            for file_name in file_name_list:
                file_path = os.path.join(dir_path, file_name)
                self.data.append(self.RawDataTuple(path=file_path, label=int(dir_name)))

        self.data.sort()
        print("\tTotal number of data = %d" % len(self.data))

        print("...Loading done.")
        return

    def reset(self):
        self.cur_idx = 0
        np.random.seed(self.epoch_counter)
        self.perm_idx = np.random.permutation(len(self.data))
        self.epoch_counter += 1
        return

    def get_empty_batch(self, batch_size):
        batch = BatchTuple(
            images=np.zeros(dtype=np.uint8, shape=[batch_size, self.image_info['height'], self.image_info['width'], self.image_info['channel']]),
            labels=np.zeros(dtype=np.int32, shape=[batch_size])
        )
        return batch

    def get_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if (self.cur_idx + batch_size) > len(self.data):
            # print('reached the end of this data set')
            self.reset()
            return None

        batch = self.get_empty_batch(batch_size)
        for idx in range(batch_size):
            single_data = self.data[self.perm_idx[self.cur_idx + idx]]

            if False:  # DEPRECATED very slow. don't use this routine.
                with tf.gfile.FastGFile(single_data.path, 'rb') as f:
                    image_data = f.read()
                decode_jpeg_tensor = tf.image.decode_jpeg(image_data, channels=3)
                image = self.sess.run(decode_jpeg_tensor)
            else:
                image = cv2.imread(single_data.path, 1)

            batch.images[idx, :, :, :] = image
            batch.labels[idx] = single_data.label

            # Verifying batch
            # print(single_data.path)
            # print(batch.images[idx, 0, 0, 0])
            # print(batch.labels[idx])

        self.cur_idx += batch_size

        return batch


class Cifar10Loader(Loader):
    label_name = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]


class OcrLoader(Loader):
    label_name = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9"
    ]






