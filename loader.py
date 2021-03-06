import os
import glob
import collections
import cv2
import numpy as np
import tensorflow as tf

BatchTuple = collections.namedtuple("BatchTuple", ['images', 'labels'])


class Loader:
    RawDataTuple = collections.namedtuple("RawDataTuple", ['path', 'label'])

    def __init__(self, data_path, image_info, batch_size):
        self.sess = tf.Session()
        self.image_info = image_info

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
            images=np.zeros(dtype=np.uint8, shape=[batch_size, self.image_info.h, self.image_info.w, self.image_info.c]),
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
            try:
                image = cv2.imread(single_data.path, 1)

                # CelebA image size : 218 x 178
                if image.shape[0] == 218 and image.shape[1] == 178:
                    image = image[20:-20, :, :]
                    image = cv2.resize(
                        image,
                        (self.image_info.h, self.image_info.w),
                        interpolation=cv2.INTER_CUBIC
                    )

                batch.images[idx, :, :, :] = image
                batch.labels[idx] = single_data.label

            except:
                print('failed to load image from [%s]... skip this batch.' % single_data.path)
                self.cur_idx += batch_size
                return None

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






