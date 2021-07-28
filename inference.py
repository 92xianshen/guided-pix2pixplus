# -*- utf-8 -*-

"""
    For uint16
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize as imresize

input_height, input_width = 512, 512
batch_size = 1
model_path = 'output/generator_g/1/' # SavedModel from training, or give by yourself
image_path = '<your own test set>' # Path to test set
result_root = '<result path>' # Path to results

def _est_ale(hazy):
    hazy_shape = tf.shape(hazy)
    yuv = tf.image.rgb_to_yuv(hazy)

    yuv_flat = tf.reshape(yuv[..., 0], [-1])
    idx = tf.argmax(yuv_flat)
    ale = tf.reshape(hazy, [-1, hazy_shape[-1]])[idx]
    ale = ale[tf.newaxis, tf.newaxis, tf.newaxis, ...]

    return ale

def normalize(im):
    return (im - im.min()) / (im.max() - im.min())

def inference():
    im_names = os.listdir(image_path)
    model = tf.saved_model.load(model_path)

    for name in im_names:
        # ==== Load and resize ====
        im_original = Image.open(os.path.join(image_path, name))
        h_original, w_original = im_original.height, im_original.width
        image = np.asarray(im_original.resize((input_width, input_height)))
        
        result_path = os.path.join(
            result_root,
            os.path.splitext(name)[0]
        )
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # model takes image of 0 ... 255
        image = image.astype(np.float32) / 255.
        images = image[np.newaxis, ...]

        ale = _est_ale(images)
        dehazy, rtme, dehazy0, tme = model([images * 255., ale * 255.], training=False)

        hazy = image
        ale = ale.numpy()[0]
        dehazy, rtme, dehazy0, tme = dehazy.numpy()[0], rtme.numpy()[
            0], dehazy0.numpy()[0], tme.numpy()[0]

        hazy = imresize(hazy, (h_original, w_original))
        dehazy = imresize(dehazy, (h_original, w_original))
        rtme = imresize(rtme, (h_original, w_original))
        dehazy0 = imresize(dehazy0, (h_original, w_original))
        tme = imresize(tme, (h_original, w_original))

        np.savez(
            os.path.join(result_path, 'out.npz'),
            hazy=hazy,
            ale=ale,
            dehazy=dehazy,
            rtme=rtme,
            dehazy0=dehazy0,
            tme=tme
        )

        dehazy, dehazy0 = normalize(dehazy), normalize(dehazy0)
        ale = np.broadcast_to(ale, hazy.shape)
        rtme, tme = np.broadcast_to(
            rtme, hazy.shape), np.broadcast_to(tme, hazy.shape)
        rtme = normalize(rtme)
        tme = normalize(tme)

        imgs = [
            hazy, ale, 
            dehazy, rtme, 
            dehazy0, tme, 
        ]
        titles = [
            'hazy.png', 'ale.png', 
            'dehazy.png', 'rtme.png', 
            'dehazy0.png', 'tme.png',
        ]

        for img, title in zip(imgs, titles):
            img = imresize(img, (h_original, w_original))
            plt.imsave(
                os.path.join(result_path, title),
                img
            )
        print('{} dehazed.'.format(name))


if __name__ == "__main__":
    inference()
