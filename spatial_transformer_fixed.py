"""
Fixed Implementation of Spatial Transformer Networks for TensorFlow 2.x / Keras 3.x

Converted to use proper Keras Layer API for compatibility with modern TensorFlow versions.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class ProjectiveTransformerLayer(Layer):
    """Spatial Projective Transformer Layer as a proper Keras Layer

    Compatible with TensorFlow 2.x and Keras 3.x functional model building.
    """

    def __init__(self, ref_img, out_size, name='SpatialProjectiveTransformer',
                 interp_method='bilinear', **kwargs):
        """
        Parameters
        ----------
        ref_img : tensor
            Reference image to transform, shape [1, height, width, channels]
        out_size : tuple of two ints
            The size of the output (height, width).
        name : string
            The layer name.
        interp_method : string
            Interpolation method: 'bilinear', 'bicubic', or 'nearest'
        """
        super(ProjectiveTransformerLayer, self).__init__(name=name, **kwargs)
        self.ref_img = ref_img
        self.out_size = out_size
        self.param_dim = 8
        self.interp_method = interp_method

    def build(self, input_shape):
        """Build the layer - create the pixel grid"""
        self.pixel_grid = _meshgrid(self.out_size)
        super(ProjectiveTransformerLayer, self).build(input_shape)

    def call(self, theta):
        """
        Projective Transformation with parameters theta

        Parameters
        ----------
        theta: float tensor
            Transformation parameters, shape [batch_size, 8].

        Returns
        -------
        output: transformed reference image
        """
        # Get batch size from theta
        batch_size = tf.shape(theta)[0]

        # Compute transformed coordinates
        x_s, y_s = self._transform(theta, batch_size)

        # Tile ref_img to batch size
        ref_img_batch = tf.tile(self.ref_img, [batch_size, 1, 1, 1])

        # Interpolate
        output = _interpolate(
            ref_img_batch, x_s, y_s,
            self.out_size,
            method=self.interp_method
        )

        # Get number of channels from ref_img
        num_channels = tf.shape(self.ref_img)[-1]

        # Reshape output
        output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], num_channels])

        return output

    def _transform(self, theta, batch_size):
        """Apply projective transformation to pixel grid"""

        # Reshape theta to (batch_size, 8) and add 9th element (1.0)
        theta = tf.reshape(theta, (batch_size, 8))
        ones = tf.ones([batch_size, 1])
        theta = tf.concat([theta, ones], 1)
        theta = tf.reshape(theta, (batch_size, 3, 3))

        # Tile pixel grid for batch
        pixel_grid_batch = tf.tile(self.pixel_grid, [batch_size])
        pixel_grid_batch = tf.reshape(pixel_grid_batch, [batch_size, 3, -1])

        # Transform: A x (x_t, y_t, 1)^T -> (x_s, y_s, z_s)
        T_g = tf.matmul(theta, pixel_grid_batch)

        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])

        # Avoid division by zero
        z_s = z_s + 1e-7

        # Normalize by homogeneous coordinate
        x_s = x_s / z_s
        y_s = y_s / z_s

        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        return x_s_flat, y_s_flat

    def get_config(self):
        config = super(ProjectiveTransformerLayer, self).get_config()
        config.update({
            'out_size': self.out_size,
            'param_dim': self.param_dim,
            'interp_method': self.interp_method
        })
        return config


def _meshgrid(out_size):
    """Create regular grid of coordinates"""
    x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0, out_size[1]),
                           tf.linspace(-1.0, 1.0, out_size[0]))
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))

    grid = tf.concat([x_t_flat, y_t_flat, tf.ones_like(x_t_flat)], 0)
    grid = tf.reshape(grid, [-1])

    return grid


def _repeat(x, n_repeats):
    """Repeat elements of a tensor"""
    rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
    return tf.reshape(rep, [-1])


def _interpolate(im, x, y, out_size, method):
    """Dispatch to appropriate interpolation method"""
    if method == 'bilinear':
        return bilinear_interp(im, x, y, out_size)
    elif method == 'bicubic':
        return bicubic_interp(im, x, y, out_size)
    elif method == 'nearest':
        return nearest_interp(im, x, y, out_size)
    return None


def bilinear_interp(im, x, y, out_size):
    """Bilinear interpolation"""
    batch_size = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    channels = tf.shape(im)[3]

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    out_height = out_size[0]
    out_width = out_size[1]

    # Clip coordinates to valid range
    x = tf.clip_by_value(x, -1, 1)
    y = tf.clip_by_value(y, -1, 1)

    # Scale from [-1, 1] to [0, width/height - 1]
    x = (x + 1.0) / 2.0 * (width_f - 1.0)
    y = (y + 1.0) / 2.0 * (height_f - 1.0)

    # Get corner points
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)
    x1_f = x0_f + 1
    y1_f = y0_f + 1

    x0 = tf.cast(x0_f, tf.int32)
    y0 = tf.cast(y0_f, tf.int32)
    x1 = tf.cast(tf.minimum(x1_f, width_f - 1), tf.int32)
    y1 = tf.cast(tf.minimum(y1_f, height_f - 1), tf.int32)

    dim2 = width
    dim1 = width * height

    base = _repeat(tf.range(batch_size) * dim1, out_height * out_width)

    base_y0 = base + y0 * dim2
    base_y1 = base + y1 * dim2

    idx_00 = base_y0 + x0
    idx_01 = base_y0 + x1
    idx_10 = base_y1 + x0
    idx_11 = base_y1 + x1

    # Gather pixel values
    im_flat = tf.reshape(im, [-1, channels])

    I00 = tf.gather(im_flat, idx_00)
    I01 = tf.gather(im_flat, idx_01)
    I10 = tf.gather(im_flat, idx_10)
    I11 = tf.gather(im_flat, idx_11)

    # Calculate weights
    w00 = tf.expand_dims(((x1_f - x) * (y1_f - y)), 1)
    w01 = tf.expand_dims(((x - x0_f) * (y1_f - y)), 1)
    w10 = tf.expand_dims(((x1_f - x) * (y - y0_f)), 1)
    w11 = tf.expand_dims(((x - x0_f) * (y - y0_f)), 1)

    output = tf.add_n([w00 * I00, w01 * I01, w10 * I10, w11 * I11])
    return output


def bicubic_interp(im, x, y, out_size):
    """Bicubic interpolation (placeholder - uses bilinear for now)"""
    # Bicubic is complex - using bilinear as fallback
    return bilinear_interp(im, x, y, out_size)


def nearest_interp(im, x, y, out_size):
    """Nearest neighbor interpolation"""
    batch_size = tf.shape(im)[0]
    height = tf.shape(im)[1]
    width = tf.shape(im)[2]
    channels = tf.shape(im)[3]

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    height_f = tf.cast(height, tf.float32)
    width_f = tf.cast(width, tf.float32)
    out_height = out_size[0]
    out_width = out_size[1]

    # Clip and scale coordinates
    x = tf.clip_by_value(x, -1, 1)
    y = tf.clip_by_value(y, -1, 1)
    x = (x + 1.0) / 2.0 * (width_f - 1.0)
    y = (y + 1.0) / 2.0 * (height_f - 1.0)

    # Round to nearest
    x0_f = tf.floor(x)
    y0_f = tf.floor(y)

    x0 = tf.cast(x0_f, tf.int32)
    y0 = tf.cast(y0_f, tf.int32)

    dim2 = width
    dim1 = width * height

    base = _repeat(tf.range(batch_size) * dim1, out_height * out_width)
    base_y0 = base + y0 * dim2
    idx = base_y0 + x0

    # Gather pixel values
    im_flat = tf.reshape(im, [-1, channels])
    output = tf.gather(im_flat, idx)

    return output