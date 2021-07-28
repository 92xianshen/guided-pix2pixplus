import tensorflow as tf
from pprint import PrettyPrinter

pp = PrettyPrinter()

# Hyperparameter
r, eps = 60, .01
clip_std = 1.5

class InstanceNormalization(tf.keras.layers.Layer):
    """ Instance Normalization Layer (https://arxiv.org/abs/1607.08022). """

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', 
            shape=input_shape[-1:], 
            initializer=tf.random_normal_initializer(1., 0.02), 
            trainable=True
        )

        self.offset = self.add_weight(
            name='offset', 
            shape=input_shape[-1:], 
            initializer='zeros', 
            trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
    """ Downsamples an input.

    Conv2D => Batchnorm => LeakyReLU

    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_norm: If True, adds the batchnorm layer

    Returns:
        Downsample Sequential Model
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters, size, strides=2, padding='same', 
            kernel_initializer=initializer, use_bias=False
        )
    )
    
    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False, name=None):
    """ Upsamples an input.

    Conv2DTranspose => Batchnorm => Dropout => ReLU

    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'
        apply_dropout: If True, adds the dropout layer

    Returns:
        Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2, 
            padding='same', kernel_initializer=initializer, use_bias=False
        )
    )

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())

    return result

class GuidedFilter(tf.keras.layers.Layer):
    """ A layer implementing guided filter """
    
    def __init__(self):
        super(GuidedFilter, self).__init__()

    def call(self, I, p, r, eps):
        def diff_x(inputs, r):
            assert inputs.shape.ndims == 4

            left    = inputs[:,         r:2 * r + 1]
            middle  = inputs[:, 2 * r + 1:         ] - inputs[:,           :-2 * r - 1]
            right   = inputs[:,        -1:         ] - inputs[:, -2 * r - 1:    -r - 1]

            outputs = tf.concat([left, middle, right], axis=1)

            return outputs

        def diff_y(inputs, r):
            assert inputs.shape.ndims == 4

            left    = inputs[:, :,         r:2 * r + 1]
            middle  = inputs[:, :, 2 * r + 1:         ] - inputs[:, :,           :-2 * r - 1]
            right   = inputs[:, :,        -1:         ] - inputs[:, :, -2 * r - 1:    -r - 1]

            outputs = tf.concat([left, middle, right], axis=2)

            return outputs

        def box_filter(x, r):
            assert x.shape.ndims == 4

            return diff_y(tf.cumsum(diff_x(tf.cumsum(x, axis=1), r), axis=2), r)

        assert I.shape.ndims == 4 and p.shape.ndims == 4

        I_shape = tf.shape(I)
        p_shape = tf.shape(p)

        # N
        N = box_filter(tf.ones((1, I_shape[1], I_shape[2], 1), dtype=I.dtype), r)

        # mean_x
        mean_I = box_filter(I, r) / N
        # mean_y
        mean_p = box_filter(p, r) / N
        # cov_xy
        cov_Ip = box_filter(I * p, r) / N - mean_I * mean_p
        # var_x
        var_I = box_filter(I * I, r) / N - mean_I * mean_I

        # A
        A = cov_Ip / (var_I + eps)
        # b
        b = mean_p - A * mean_I

        mean_A = box_filter(A, r) / N
        mean_b = box_filter(b, r) / N

        q = mean_A * I + mean_b

        return q

def standardization_clip(tensor):
    # assert tf.less_equal(tf.shape(tensor)[0], 1) # batch_size should be 1.
    mean, std = tf.math.reduce_mean(tensor), tf.math.reduce_std(tensor)
    tensor_norm = (tensor - mean) / std
    return tf.clip_by_value(
        tensor_norm, clip_value_min=-clip_std, clip_value_max=clip_std)

def create_generator(input_shape=(512, 512, 3), output_shape=(512, 512, 3), norm_type='batchnorm'):
    # Backbone
    backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    I0 = backbone.input
    
    # Reuse features
    layer_names = [
        'block1a_project_conv', # (None, 256, 256, 16)
        'block2b_project_conv', # (None, 128, 128, 24)
        'block3b_project_conv', # (None, 64, 64, 40)
        'block5c_project_conv', # (None, 32, 32, 112)
    ]
    features = [backbone.get_layer(name).output for name in layer_names]
    features = reversed(features)
    
    # Upsampling
    up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True, name='upsample1'), # (None, 32, 32, 512 + 112)
        upsample(256, 4, norm_type, name='upsample2'), # (None, 64, 64, 256 + 40)
        upsample(128, 4, norm_type, name='upsample3'), # (None, 128, 128, 128 + 24)
        upsample(64, 4, norm_type, name='upsample4'), # (None, 256, 256, 64 + 16)
    ]

    # Last
    last = tf.keras.layers.Conv2DTranspose(
        output_shape[-1], 4, strides=2, padding='same', 
        kernel_initializer=tf.random_normal_initializer(0., .02), 
        activation='linear'
    )

    # Guided filter
    guided_filter = GuidedFilter()

    # Forward
    x = backbone.output
    for up, feat in zip(up_stack, features):
        x = up(x)
        x = tf.keras.layers.concatenate([x, feat])
    x = last(x)

    # Input, note that I0 and A0 are 0...255
    A0 = tf.keras.Input(shape=[None, None, input_shape[-1]], name='AtmosphericLight')
    I, A = I0 / 255., A0 / 255.
    
    # Refine
    betadepth0 = x
    guidance = tf.image.rgb_to_grayscale(I)
    betadepth = guided_filter(guidance, betadepth0, r=r, eps=eps)

    # t
    t0 = tf.math.exp(-betadepth0)
    t = tf.math.exp(-betadepth)

    J0 = (I - A) / t0 + A
    J = (I - A) / t + A

    # Standardization clip
    J0 = standardization_clip(J0)
    J = standardization_clip(J)

    return tf.keras.Model(inputs=[I0, A0], outputs=[J, t, J0, t0])

def create_discriminator(input_channels, norm_type='batchnorm', target=True):
    """ PatchGAN discriminator model (https://arxiv.org/abs/1611.07004).

    Args:
        input_channels: Input channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
        target: Bool, indicating whether target image is an input or not.

    Returns:
        Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, input_channels], name='input_image')
    x = inp

    if target:
        tar = tf.keras.layers.Input(shape=[None, None, input_channels], name='target_image')
        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 512, 512, channels * 2)

    down1 = downsample(64, 4, norm_type, False)(x) # (bs, 256, 256, 64)
    down2 = downsample(128, 4, norm_type)(down1) # (bs, 128, 128, 128)
    down3 = downsample(256, 4, norm_type)(down2) # (bs, 64, 64, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 66, 66, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, 
        use_bias=False
    )(zero_pad1) # (bs, 63, 63, 512)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = InstanceNormalization()(conv)
    
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 64, 64, 512)

    last = tf.keras.layers.Conv2D(
        1, 4, strides=1, 
        kernel_initializer=initializer
    )(zero_pad2) # (bs, 61, 61, 1)

    if target:
        return tf.keras.Model(inputs=[inp, tar], outputs=last)
    else:
        return tf.keras.Model(inputs=inp, outputs=last)

        
if __name__ == '__main__':
    g = create_generator()
    g.summary()

    d = create_discriminator(3)
    d.summary()