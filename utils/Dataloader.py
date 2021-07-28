import os
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image

def _byte_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value_list):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image2tfrecord(image_path, label_path, output_path):
    """ Convert images and labels into TFRecord """

    image_names = os.listdir(image_path)
    tfrecord_path = os.path.join(output_path, 'tfrecord/')
    if os.path.exists(tfrecord_path):
        shutil.rmtree(tfrecord_path)
    os.makedirs(tfrecord_path)

    for im_name in image_names:
        print('Converting {}...'.format(im_name))
        im_name_prefix = os.path.splitext(im_name)[0]
        try:
            image = np.asarray(Image.open(
                os.path.join(image_path, im_name)))
            label = np.asarray(Image.open(
                os.path.join(label_path, im_name)))
            writer = tf.io.TFRecordWriter(
                os.path.join(tfrecord_path, im_name_prefix + '.tfrecord'))
        except:
            print('File {} open error!'.format(im_name))
            continue

        assert image.shape == label.shape and len(image.shape) == 3
        height, width, num_channels = image.shape

        image = image.tostring()
        label = label.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _byte_feature(image),
            'label': _byte_feature(label),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'num_channels': _int64_feature(num_channels)
        }))

        writer.write(example.SerializeToString())
        writer.close()
        print('{} converted.'.format(im_name))


def load_dataset(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'num_channels': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(
            example_proto, feature_description)

        image = tf.io.decode_raw(example['image'], tf.uint8)
        label = tf.io.decode_raw(example['label'], tf.uint8)
        height, width, num_channels = example['height'], example['width'], example['num_channels']

        image = tf.reshape(image, [height, width, num_channels])
        label = tf.reshape(label, [height, width, num_channels])

        image = tf.cast(image, tf.float32) / 255.
        label = tf.cast(label, tf.float32) / 255.

        example['image'] = image
        example['label'] = label

        return example

    dataset = dataset.map(_parse_function).batch(batch_size)

    return dataset
