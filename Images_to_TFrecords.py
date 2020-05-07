# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle
from data_augmentation import *

_available_augmentation = {
    mean_white_balance, perfect_reflective_white_balance, gray_world_assumes_white_balance,
    color_correction_of_image_analysis, dynamic_threshold_white_balance, gamma_trans,
    contrast_image_correction, flip_img
}


def image_path_load(train_path, classes, is_shuffle=True):
    """"加载数据，数据文件夹为
    --trainset
        --classname1
            --file1
            --file2
            ....
        --classname2
        ....

    :param train_path(str):数据路径
    :param image_size(int):图像大小
    :param classes:类别
    :param is_augmentation(bool):是否进行数据扩增

    :return images:图像集 path
    :return labels:标签集
    """
    images_path = []
    labels = []
    print('Reading training images')
    for fld in classes:  # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld)
        files = os.listdir(path)
        for file in tqdm(files):
            read_number = 0
            image_file_path = os.path.join(path, file)
            second_files = os.listdir(image_file_path)
            for fl in second_files:
                if fl[-4:] != ".jpg" and fl[-4:] != ".png":
                    continue
                read_number += 1

                image_path = os.path.join(image_file_path, fl)
                images_path.append(image_path)
                label = index
                labels.append(label)
    if is_shuffle:
        images_path, labels = shuffle(images_path, labels)
    labels = np.array(labels)
    return images_path, labels


def image_to_tfrecord(image_path_list, labels, tfrecord_save_path, tfrecord_image_number, is_augmentation=False):
    """
    :param image_path: image path list
    :param tfrecord_save_path: tfrecord save path
    :param tfrecord_image_number:per tfrecord save image number
    """

    # 存放图片个数
    bestnum = tfrecord_image_number
    # 第几个图片
    num = 0
    # 第几个TFRecord文件
    recordfilenum = 0

    if not os.path.exists(tfrecord_save_path):
        os.mkdir(tfrecord_save_path)

    ftrecordfilename = ("traindata.tfrecords-%.4d" % recordfilenum)
    writer = tf.io.TFRecordWriter(os.path.join(tfrecord_save_path, ftrecordfilename))
    # 类别和路径

    for image_path, label in tqdm(zip(image_path_list, labels)):
        num = num + 1
        if num > bestnum:
            num = 1
            recordfilenum = recordfilenum + 1
            # tfrecords格式文件名
            writer.close()
            ftrecordfilename = ("traindata.tfrecords-%.4d" % recordfilenum)
            writer = tf.io.TFRecordWriter(os.path.join(tfrecord_save_path, ftrecordfilename))

        # img = Image.open(image_path, 'r')
        # print(image_path)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        size = img.shape

        if is_augmentation:
            for augmentation_type in _available_augmentation:
                try:
                    img_augmentation = augmentation_type(img)
                    img_raw = img_augmentation.tobytes()  # 将图片转化为二进制格式
                    example = tf.train.Example(
                        features=tf.train.Features(feature={
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                            'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
                        }))
                    writer.write(example.SerializeToString())  # 序列化为字符串
                    num = num + 1
                    if num > bestnum:
                        num = 1
                        recordfilenum = recordfilenum + 1
                        # tfrecords格式文件名
                        writer.close()
                        ftrecordfilename = ("traindata.tfrecords-%.4d" % recordfilenum)
                        writer = tf.io.TFRecordWriter(os.path.join(tfrecord_save_path, ftrecordfilename))
                except:
                    pass

        img_raw = img.tobytes()  # 将图片转化为二进制格式
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
            }))
        writer.write(example.SerializeToString())  # 序列化为字符串
    writer.close()


if __name__ == '__main__':
    datasets = ['trainset', 'testset']
    # 图片路径
    data_path = 'data'
    # 文件路径
    tfrecord_save_path = 'tfrecord'
    is_augmentation = True
    if not os.path.exists(tfrecord_save_path):
        os.mkdir(tfrecord_save_path)
    # 存放图片个数
    tfrecord_image_number = 100
    # 类别  classe
    classes = ['ImposterFace',
               'ClientFace']
    for dataset in datasets:
        image_path = os.path.join(data_path, dataset)

        images_path_list, labels = image_path_load(image_path, classes)
        # tfrecords格式文件名
        save_paths = os.path.join(tfrecord_save_path, dataset)
        image_to_tfrecord(images_path_list, labels, save_paths, tfrecord_image_number, is_augmentation)
