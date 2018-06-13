import tensorflow as tf
from helper.helper_api import Helper
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

class Deep3d:
    def __init__(self):
        self.helber = Helper()

    def add_activation_layer(self, input, name):
        layer = tf.nn.relu(features=input, name=name)
        return layer

    def add_selection_layer(self, masks, left_image, left_shift=16, name="select"):
        _, H, W, S = masks.get_shape().as_list()
        print(H,W)
        with tf.variable_scope(name):
            padded = tf.pad(left_image, [[0, 0], [0, 0], [left_shift, left_shift], [0, 0]], mode='REFLECT')

            # padded is the image padded whatever the left_shift variable is on either side
            layers = []
            for s in np.arange(S):
                layers.append(tf.slice(padded, [0, 0, s, 0], [-1, 160, 384, -1]))

            slices = tf.stack(layers, axis=4)
            disparity_image = tf.multiply(slices, tf.expand_dims(masks, axis=3))

            return tf.reduce_sum(disparity_image, axis=4)

    def build_model(self, input_rgb):
        # group 1
        self.conv1_1 = self.helber.add_convolutional_layer(input=input_rgb, name="conv1_1", padding=(1, 1))
        self.relu1_1 = self.add_activation_layer(input=self.conv1_1, name="relu1_1")
        self.pool1 = self.helber.add_pooling_layer(input=self.relu1_1, name="pool1")
        # group 2
        self.conv2_1 = self.helber.add_convolutional_layer(input=self.pool1, name="conv2_1", padding=(1, 1))
        self.relu2_1 = self.add_activation_layer(input=self.conv2_1, name="relu2_1")
        self.pool2 = self.helber.add_pooling_layer(input=self.relu2_1, name="pool2")

        # group 3
        self.conv3_1 = self.helber.add_convolutional_layer(input=self.pool2, name="conv3_1", padding=(1, 1))
        self.relu3_1 = self.add_activation_layer(input=self.conv3_1, name="relu3_1")
        self.conv3_2 = self.helber.add_convolutional_layer(input=self.relu3_1, name="conv3_2", padding=(1, 1))
        self.relu3_2 = self.add_activation_layer(input=self.conv3_2, name="relu3_2")
        self.pool3 = self.helber.add_pooling_layer(input=self.relu3_2, name="pool3")

        # group 4
        self.conv4_1 = self.helber.add_convolutional_layer(input=self.pool3, name="conv4_1", padding=(1, 1))
        self.relu4_1 = self.add_activation_layer(input=self.conv4_1, name="relu4_1")
        self.conv4_2 = self.helber.add_convolutional_layer(input=self.relu4_1, name="conv4_2", padding=(1, 1))
        self.relu4_2 = self.add_activation_layer(input=self.conv4_2, name="relu3_2")
        self.pool4 = self.helber.add_pooling_layer(input=self.relu4_2, name="pool4")

        # group 5
        self.conv5_1 = self.helber.add_convolutional_layer(input=self.pool4, name="conv5_1", padding=(1, 1))
        self.relu5_1 = self.add_activation_layer(input=self.conv5_1, name="relu5_1")
        self.conv5_2 = self.helber.add_convolutional_layer(input=self.relu5_1, name="conv5_2", padding=(1, 1))
        self.relu5_2 = self.add_activation_layer(input=self.conv5_2, name="relu5_2")
        self.pool5 = self.helber.add_pooling_layer(input=self.relu5_2, name="pool5")

        # group 6
        self.flatten = tf.contrib.layers.flatten(self.pool5)
        self.fc6 = self.helber.add_fully_connected(input=self.flatten, name="fc6")
        self.relu6 = self.add_activation_layer(input=self.fc6, name="relu6")

        # group 7
        self.fc7 = self.helber.add_fully_connected(input=self.relu6, name="fc7")
        self.relu7 = self.add_activation_layer(input=self.fc6, name="relu7")

        # output
        self.fc8 = self.helber.add_fully_connected(input=self.relu7, name="pred5")
        self.pred5 = tf.reshape(self.fc8, shape=[-1,5, 12, 33])

        # Branch 4
        self.bn_pool4 = self.helber.add_batch_normalization_layer(input=self.pool4, name='bn_pool4')
        self.pred4 = self.helber.add_convolutional_layer(input=self.bn_pool4, name='pred4', padding=(1, 1))
        # Branch 3
        self.bn_pool3 = self.helber.add_batch_normalization_layer(input=self.pool3, name='bn_pool3')
        self.pred3 = self.helber.add_convolutional_layer(input=self.bn_pool3, name='pred3', padding=(1, 1))
        # Branch 2
        self.bn_pool2 = self.helber.add_batch_normalization_layer(input=self.pool2, name='bn_pool2')
        self.pred2 = self.helber.add_convolutional_layer(input=self.bn_pool2, name='pred2', padding=(1, 1))
        # Branch 1
        self.bn_pool1 = self.helber.add_batch_normalization_layer(input=self.pool1, name='bn_pool1')
        self.pred1 = self.helber.add_convolutional_layer(input=self.bn_pool1, name='pred1', padding=(1, 1))

        scale = 1
        self.relu_pred1 = self.add_activation_layer(input=self.pred1, name='relu_pred1')
        self.deconv_pred1 = self.helber.add_deconvolutional_layer(input=self.relu_pred1, scale=scale,name='deconv_pred1')

        scale *= 2
        self.relu_pred2 = self.add_activation_layer(input=self.pred2, name='relu_pred2')
        self.deconv_pred2 = self.helber.add_deconvolutional_layer(input=self.relu_pred2, scale=scale,
                                                                  name='deconv_pred2')

        scale *= 2
        self.relu_pred3 = self.add_activation_layer(input=self.pred3, name='relu_pred3')
        self.deconv_pred3 = self.helber.add_deconvolutional_layer(input=self.relu_pred3, scale=scale,
                                                                  name='deconv_pred3')

        scale *= 2
        self.relu_pred4 = self.add_activation_layer(input=self.pred4, name='relu_pred4')
        self.deconv_pred4 = self.helber.add_deconvolutional_layer(input=self.relu_pred4, scale=scale,
                                                                  name='deconv_pred4')

        scale *= 2
        self.relu_pred5 = self.add_activation_layer(input=self.pred5, name='relu_pred5')
        self.deconv_pred5 = self.helber.add_deconvolutional_layer(input=self.relu_pred5, scale=scale,
                                                                  name='deconv_pred5')

        # merge
        print(self.deconv_pred1.shape)
        self.feat = self.deconv_pred1 + self.deconv_pred2 + self.deconv_pred3 + self.deconv_pred4 + self.deconv_pred5
        self.feat_act = self.add_activation_layer(input=self.feat, name='feat_act')
        scale = 2
        self.deconv_predup = self.helber.add_deconvolutional_layer(input=self.feat_act, scale=scale,
                                                                   name='deconv_predup')
        self.relu_predup = self.add_activation_layer(input=self.deconv_predup, name='relu_predup')
        self.conv_predup = self.helber.add_convolutional_layer(input=self.relu_predup, name="convolution0",
                                                               padding=(1, 1))

        # selection
        self.mask = tf.nn.softmax(self.conv_predup)
        self.right_image = self.add_selection_layer(masks=self.mask, left_image=input_rgb)

        return self.right_image
