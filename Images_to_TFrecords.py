# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 23:53:56 2017

@author: zhangxu
"""
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

n_classes = 2
feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
    'label': tf.FixedLenFeature([], tf.int64),
    'img_raw': tf.FixedLenFeature([], tf.string),
    'img_width': tf.FixedLenFeature([], tf.int64),
    'img_height': tf.FixedLenFeature([], tf.int64)
}


def read_and_decode(example_string):
    '''
    从TFrecord格式文件中读取数据
    '''

    features = tf.parse_single_example(example_string,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'img_width': tf.FixedLenFeature([], tf.int64),
                                           'img_height': tf.FixedLenFeature([], tf.int64)
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int64)
    label = tf.one_hot(label, n_classes)
    return img, label


def get_dataset_batch(data_files, batch_size=4):
    dataset = tf.data.TFRecordDataset(data_files)
    dataset = dataset.repeat()  # 重复数据集
    dataset = dataset.map(read_and_decode)  # 解析数据
    dataset = dataset.shuffle(buffer_size=100)  # 在缓冲区中随机打乱数据
    batch = dataset.batch(batch_size=batch_size)  # 每10条数据为一个batch，生成一个新的Datasets
    return batch


def get_model():
    model = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = model(inputs)  # 此处x为MobileNetV2模型去处顶层时输出的特征相应图。
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax',
                                    use_bias=True, name='Logits')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def train(model, batch):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(batch, epochs=10, steps_per_epoch=1000)
    return model


def model_save(model, save_path):
    # tf.enable_eager_execution()
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=save_path,
                      name="frozen_graph.pb",
                      as_text=False)


if __name__ == "__main__":
    # TFRecord文件路径
    # tf.enable_eager_execution()
    data_path = 'tfrecord/testset/*'
    save_path = "model/frozen_graph.h5"
    batch_size = 4
    # 获取文件名列表
    data_files = tf.gfile.Glob(data_path)
    # print(data_files)
    print("************get dataset batch ***********")
    batch = get_dataset_batch(data_files, batch_size=batch_size)
    model = get_model()
    print("************gstart train ***********")
    model = train(model, batch)
    model.save(save_path)

# 文件名列表生成器

# def read_and_decode(filename, n_classes=2, batch_size=8):
#     filename_queue = tf.train.string_input_producer(filename)
#
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example,
#                                        features={
#                                            'label': tf.FixedLenFeature([], tf.int64),
#                                            'img_raw': tf.FixedLenFeature([], tf.string),
#                                            'img_width': tf.FixedLenFeature([], tf.int64),
#                                            'img_height': tf.FixedLenFeature([], tf.int64)
#                                        })
#
#     img = tf.decode_raw(features['img_raw'], tf.uint8)
#     img = tf.reshape(img, [224, 224, 3])
#     img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
#     label = tf.cast(features['label'], tf.int64)
#
#     image_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                       batch_size=batch_size,
#                                                       capacity=2000,
#                                                       min_after_dequeue=1000)
#
#     label_batch = tf.one_hot(label_batch, n_classes)
#     label_batch = tf.cast(label_batch, dtype=tf.int64)
#     label_batch = tf.reshape(label_batch, [batch_size, n_classes])
#
#     dataset = reader.map(image_batch, label_batch)
#     dataset = reader.batch(batch_size)
#     iterator = dataset.make_one_shot_iterator()
#     skeleton, label = iterator.get_next()
#     return skeleton, label
#
#
#
# x_train, y_train = read_and_decode(data_files)


# raw_dataset = tf.data.TFRecordDataset(data_files)
# model = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
# inputs = tf.keras.layers.Input(shape=(224, 224, 3))
# x = model(inputs)  # 此处x为MobileNetV2模型去处顶层时输出的特征相应图。
# x = tf.keras.layers.GlobalAveragePooling2D()(x)
# outputs = tf.keras.layers.Dense(2, activation='softmax',
#                                 use_bias=True, name='Logits')(x)
# model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
# model.summary()
# model.compile(
#     optimizer=tf.train.AdamOptimizer(0.001),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )
#
# history = model.fit_generator(x_train, y_train,
#                               epochs=10,
#                               )
