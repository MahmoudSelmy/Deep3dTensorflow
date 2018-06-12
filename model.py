import tensorflow as tf
from helper.helper_api import Helper

class Deep3d:
    def __init__(self):
        self.helber = Helper()

    def build_model(self,input_rgb):
        #group 1
        self.conv1_1 = self.helber.add_convolutional_layer(input=input_rgb,name="conv1_1",padding=(1,1))
        self.relu1_1 = self.helper.add_activation_layer(input=self.conv1_1,name="relu1_1")
        self.pool1 = self.helber.add_pooling_layer(input=self.relu1_1,name="pool1")

        # group 2
        self.conv2_1 = self.helber.add_convolutional_layer(input=self.pool1, name="conv2_1", padding=(1, 1))
        self.relu2_1 = self.helper.add_activation_layer(input=self.conv2_1, name="relu2_1")
        self.pool2 = self.helber.add_pooling_layer(input=self.relu2_1, name="pool2")

        # group 3
        self.conv3_1 = self.helber.add_convolutional_layer(input=self.pool2, name="conv3_1", padding=(1, 1))
        self.relu3_1 = self.helper.add_activation_layer(input=self.conv3_1, name="relu3_1")
        self.conv3_2 = self.helber.add_convolutional_layer(input=self.relu3_1, name="conv3_2", padding=(1, 1))
        self.relu3_2 = self.helper.add_activation_layer(input=self.conv3_2, name="relu3_2")
        self.pool3 = self.helber.add_pooling_layer(input=self.relu3_2, name="pool3")

        # group 4
        self.conv4_1 = self.helber.add_convolutional_layer(input=self.pool3, name="conv4_1", padding=(1, 1))
        self.relu4_1 = self.helper.add_activation_layer(input=self.conv4_1, name="relu4_1")
        self.conv4_2 = self.helber.add_convolutional_layer(input=self.relu4_1, name="conv4_2", padding=(1, 1))
        self.relu4_2 = self.helper.add_activation_layer(input=self.conv4_2, name="relu3_2")
        self.pool4 = self.helber.add_pooling_layer(input=self.relu4_2, name="pool4")

        # group 5
        self.conv5_1 = self.helber.add_convolutional_layer(input=self.pool4, name="conv5_1", padding=(1, 1))
        self.relu5_1 = self.helper.add_activation_layer(input=self.conv5_1, name="relu5_1")
        self.conv5_2 = self.helber.add_convolutional_layer(input=self.relu5_1, name="conv5_2", padding=(1, 1))
        self.relu5_2 = self.helper.add_activation_layer(input=self.conv5_2, name="relu5_2")
        self.pool5 = self.helber.add_pooling_layer(input=self.relu5_2, name="pool5")

        # group 6
        self.flatten = tf.contrib.layers.flatten(self.pool5)
        self.fc6 = self.helber.add_fully_connected(input=self.flatten,name="fc6")
        self.relu6 = self.helper.add_activation_layer(input=self.fc6, name="relu6")

        # group 7
        self.fc7 = self.helber.add_fully_connected(input=self.relu6, name="fc7")
        self.relu6 = self.helper.add_activation_layer(input=self.fc6, name="relu7")

        # output
        self.fc8 = self.helber.add_fully_connected(input=self.relu7, name="pred5")

        self.pred5 = tf.reshape(self.fc8,shape=[-1,5,12,33])

        
