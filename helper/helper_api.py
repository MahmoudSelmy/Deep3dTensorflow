import tensorflow as tf
import numpy as np

class Vgg16Model:
    def __init__(self, weights_path='./weights/deep3d.npy'):
        self.weights = np.load('vgg16.npy', encoding='latin1').item()
        self.trainable = False

    def add_convolutional_layer(self,input,name,padding):
        weights_key = name + '_weight'
        biases_key = name + '_bias'

        kernel = self.weights[weights_key]
        bias = self.weights[biases_key]

        no_of_filters, height, width, channels = kernel.shape

        padded_input = tf.pad(input, [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]],'CONSTANT')

        layer = tf.layers.conv2d(padded_input, no_of_filters, kernel_size=(height, width),
                                 padding='VALID', name=name, trainable=self.trainable,
                                 kernel_initializer=tf.constant_initializer(kernel, dtype=tf.float32),
                                 bias_initializer=tf.constant_initializer(bias, dtype=tf.float32),use_bias=True)
        return layer

    def add_pooling_layer(self,input,name):
        layer = tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2, 1],padding='VALID', name=name)
        return layer

    def add_activation_layer(self,input,name):
        layer = tf.nn.relu(input,name=name)
        return layer

    def add_fully_connected(self,input,name):
        weights_key = name + '_weight'
        biases_key = name + '_bias'

        weights = self.weights[weights_key]
        bias = self.weights[biases_key]

        layer = tf.nn.xw_plus_b(input, weights, bias, name=name)

        return layer

    def add_batch_normalization_layer(self,input,name):
        beta_name = name + '_beta'
        gamma_name = name + '_gamma'
        moving_inv_var_name  = name + '_moving_inv_var'
        moving_mean_name = name + '_moving_mean'

        beta = self.weights[beta_name]
        gamma = self.weights[gamma_name]
        moving_inv_var = self.weights[moving_inv_var_name]
        moving_mean = self.weights[moving_mean_name]
        epsilon = 0.001

        layer = tf.nn.batch_normalization(input, moving_mean, moving_inv_var, beta, gamma, epsilon, name=name)

        return layer

    def add_deconvolutional_layer(self,input,scale,name):
        weights_key = name + '_weight'
        biases_key = name + '_bias'

        weights = self.weights[weights_key]
        bias = self.weights[biases_key]

        weights_var = tf.get_variable(name= weights_key,shape=weights.shape,initializer=tf.constant_initializer(weights))
        bias_var = tf.get_variable(name= biases_key,shape=bias.shape,initializer=tf.constant_initializer(bias))

        dyn_input_shape = tf.shape(input)
        N = dyn_input_shape[0]
        H = dyn_input_shape[1]
        W = dyn_input_shape[2]

        out_channels = weights.shape[0]
        shape_output = tf.stack([N,
                                 scale * (H - 1) + scale * 2 - scale,
                                 scale * (W - 1) + scale * 2 - scale,
                                 out_channels])

        deconv = tf.nn.conv2d_transpose(input, weights_var, shape_output, [1, scale, scale, 1])
        layer = tf.nn.bias_add(deconv, bias_var)

        return layer

    def add_selection_layer(self,masks, left_image, left_shift=16, name="select"):
        _, H, W, S = masks.get_shape().as_list()
        with tf.variable_scope(name):
            padded = tf.pad(left_image, [[0, 0], [0, 0], [left_shift, left_shift], [0, 0]], mode='REFLECT')

            # padded is the image padded whatever the left_shift variable is on either side
            layers = []
            for s in np.arange(S):
                layers.append(tf.slice(padded, [0, 0, s, 0], [-1, H, W, -1]))

            slices = tf.stack(layers, axis=4)
            disparity_image = tf.multiply(slices, tf.expand_dims(masks, axis=3))

            return tf.reduce_sum(disparity_image, axis=4)

