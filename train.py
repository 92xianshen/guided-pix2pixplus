# -*- utf-8 -*-

"""
    For uint16
"""
import os

import shutil
import sys
import time
from xml.dom.minidom import parse

import numpy as np
import tensorflow as tf
from PIL import Image

from model import CustomEfficientNetB0
from utils.XMLUtil import readXML
from utils.Dataloader import image2tfrecord, load_dataset

input_height, input_width = 512, 512
batch_size = 1
buffer_size = batch_size * 4
checkpoint_path = 'checkpoints/train/'
LAMBDA = 100
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def gradient_penalty(discriminator, real_image, fake_image):
    batch_size = real_image.shape[0]

    # [b, h, w, c]
    t = tf.random.uniform([batch_size, 1, 1, 1])
    # [b, 1, 1, 1] => [b, h, w, c]
    t = tf.broadcast_to(t, real_image.shape)

    interpolate = t * real_image + (1 - t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interpolate])
        d_interpolate_logits = discriminator(interpolate, training=True)
    grads = tape.gradient(d_interpolate_logits, interpolate)

    # grads: [b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((gp - 1) ** 2)

    return gp


def generator_loss(generator, disc_generated_output, gen_output, target):
    gan_loss = -tf.reduce_mean(disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    # Parameter regularization
    loss_regularizations = []
    for p in generator.trainable_variables:
        loss_regularizations.append(tf.nn.l2_loss(p))
    loss_regularization = tf.reduce_sum(tf.stack(loss_regularizations))

    total_gen_loss = gan_loss + \
        (LAMBDA * l1_loss) + .001 * loss_regularization

    return total_gen_loss, gan_loss, l1_loss, loss_regularization


def discriminator_loss(discriminator, real_image, fake_image, disc_real_output, disc_generated_output):
    d_loss_real = -tf.reduce_mean(disc_real_output)
    d_loss_fake = tf.reduce_mean(disc_generated_output)
    gp = gradient_penalty(discriminator, real_image, fake_image)

    loss = d_loss_real + d_loss_fake + 10. * gp

    return loss, gp


def train_loop(args, output_path):
    # Load train and val sets
    if not args['from_tfrecord']:
        tfrecord_path = os.path.join(output_path, 'tfrecord')
    else:
        tfrecord_path = os.path.join(input_path, 'tfrecord')
    train_names = os.listdir(tfrecord_path)
    train_names = [os.path.join(tfrecord_path, name) for name in train_names]
    train_set = load_dataset(train_names, batch_size=batch_size)

    # Log
    logdir = os.path.join(output_path, 'logs/')
    file_writer = tf.summary.create_file_writer(logdir + 'metrics')
    file_writer.set_as_default()

    if args['visualization']:
        import matplotlib.pyplot as plt
        vis_path = os.path.join(output_path, 'visualizations')
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)

    # Generator: hazy --> dehazy
    generator = CustomEfficientNetB0.create_generator(norm_type='instancenorm')
    # Discriminator: real or fake
    discriminator = CustomEfficientNetB0.create_discriminator(
        input_channels=args['input_channels'], norm_type='instancenorm', target=False)

    # Optimizers
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        args['initial_learning_rate'],
        decay_steps=args['decay_steps'],
        decay_rate=args['decay_rate']
    )

    generator_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)

    # Checkpoints and manager
    ckpt = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
    )
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, os.path.join(output_path, args['checkpoint_path']), max_to_keep=5)

    # if restoration is enabled and a checkpoint exists, restore the latest checkpoint.
    if args['restore'] and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored, from {}'.format(
            ckpt_manager.latest_checkpoint))

    def _random_crop(record):
        hazy, gt = record['image'], record['label']

        images = tf.concat([hazy, gt], axis=-1)
        images = tf.image.resize_with_pad(images, input_height + input_height // 2,
                                          input_width + input_width // 2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        images = tf.image.random_crop(
            images, [batch_size, input_height, input_width, args['input_channels'] * 2])
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)

        hazy, gt = images[..., :args['input_channels']
                          ], images[..., args['input_channels']:]

        record['image'], record['label'] = hazy, gt

        return record

    def _est_ale(hazy):
        hazy_shape = tf.shape(hazy)
        hsv = tf.image.rgb_to_hsv(hazy)
        hsv_shape = tf.shape(hsv)

        hsv = tf.reshape(hsv[..., 2], [-1])
        idx = tf.argmax(hsv)
        ale = tf.reshape(hazy, [-1, hazy_shape[-1]])[idx]
        ale = ale[tf.newaxis, tf.newaxis, tf.newaxis, ...]

        return ale

    @tf.function
    def _train_step(hazy, gt, ale):
        # hazy: 0 ... 1, gt: 0 mean, 1 std
        with tf.GradientTape(persistent=True) as tape:
            dehazy, rtme, dehazy0, tme = generator(
                [hazy * 255., ale * 255.], training=True) # 0 ... 255 --> -1.5 ... 1.5

            disc_real_output = discriminator(gt, training=True)
            disc_generated_output = discriminator(dehazy, training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss, l_reg = generator_loss(
                generator, disc_generated_output, dehazy, gt)
            disc_loss, gp = discriminator_loss(
                discriminator, gt, dehazy, disc_real_output, disc_generated_output)

        generator_gradients = tape.gradient(
            gen_total_loss, generator.trainable_variables)
        discriminator_gradients = tape.gradient(
            disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(
            zip(generator_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, discriminator.trainable_variables))

        return gen_total_loss, gen_gan_loss, gen_l1_loss, l_reg, disc_loss, gp

    def standardize(tensor):
        # assert tf.less_equal(tf.shape(tensor)[0], 1) # batch_size should be 1.
        mean, std = tf.math.reduce_mean(tensor), tf.math.reduce_std(tensor)
        return (tensor - mean) / std

    def normalize(im):
        return (im - im.min()) / (im.max() - im.min())

    with file_writer.as_default():
        step = 0
        for epoch in range(args['num_epochs']):
            start = time.time()
            train_set_ = train_set.map(_random_crop).shuffle(
                buffer_size=buffer_size)

            for record in train_set_:
                hazy, gt = record['image'], record['label']
                ale = _est_ale(hazy)
                gen_total_loss, gen_gan_loss, gen_l1_loss, l_reg, disc_loss, gp = _train_step(
                    hazy, standardize(gt), ale)
                step += 1

                tf.summary.scalar('gen_total_loss', gen_total_loss, step=step)
                tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
                tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
                tf.summary.scalar('l_reg', l_reg, step=step)
                tf.summary.scalar('disc_loss', disc_loss, step=step)
                tf.summary.scalar('gp', gp, step=step)

                print('Step {}, gen_total_loss: {}, gen_gan_loss: {}, gen_l1_loss: {}, l_reg: {}, disc_loss: {}, gp: {}'.format(
                    step, gen_total_loss.numpy(), gen_gan_loss.numpy(), gen_l1_loss.numpy(), l_reg.numpy(), disc_loss.numpy(), gp.numpy()))

            print('Time taken for epoch {} is {} sec'.format(
                epoch + 1, time.time() - start))

            ckpt_save_path = ckpt_manager.save()
            print('Checkpoint for epoch {} saved at {}'.format(
                epoch + 1, ckpt_save_path))

            # Train visualization
            if args['visualization']:
                for record in train_set_.take(1):
                    hazy, gt = record['image'], record['label']
                    ale = _est_ale(hazy)
                    dehazy, rtme, dehazy0, tme = generator(
                        [hazy * 255., ale * 255.], training=False)

                    hazy, gt, ale = hazy.numpy()[0], gt.numpy()[
                        0], ale.numpy()[0]
                    dehazy, rtme, dehazy0, tme = dehazy.numpy()[0], rtme.numpy()[
                        0], dehazy0.numpy()[0], tme.numpy()[0]

                    np.savez(os.path.join(
                        vis_path, 'epoch_{}.npz'.format(epoch)),
                        hazy=hazy,
                        gt=gt,
                        ale=ale,
                        dehazy=dehazy,
                        rtme=rtme,
                        dehazy0=dehazy0,
                        tme=tme)

                    dehazy, dehazy0 = normalize(dehazy), normalize(dehazy0)
                    ale = np.broadcast_to(ale, hazy.shape)
                    rtme, tme = np.broadcast_to(
                        rtme, hazy.shape), np.broadcast_to(tme, hazy.shape)
                    rtme = (rtme - rtme.min()) / (rtme.max() - rtme.min())
                    tme = (tme - tme.min()) / (tme.max() - tme.min())

                    imgs = [hazy, gt, ale, np.ones_like(hazy),
                            dehazy, rtme, dehazy0, tme]
                    titles = ['Hazy', 'GT', 'ALE', '',
                              'Dehazy', 'RTME', 'Hazy', 'TME']

                    row, col = 2, 4
                    plt.figure(figsize=(4 * col, 4 * row))
                    for i in range(len(imgs)):
                        plt.subplot(row, col, i + 1)
                        plt.title(titles[i])
                        plt.imshow(imgs[i])
                    plt.savefig(os.path.join(
                        vis_path, 'epoch_{}.png'.format(epoch)))
                    plt.close()

    tf.saved_model.save(generator, os.path.join(output_path, 'generator_g/1/'))


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    train_args = readXML(os.path.join(input_path, 'train.xml'))

    if not train_args['from_tfrecord']:
        image_path = os.path.join(input_path, 'cloud')
        label_path = os.path.join(input_path, 'label')
        image2tfrecord(image_path, label_path, output_path)

    train_loop(train_args, output_path)
