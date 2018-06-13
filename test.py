import tensorflow as tf
import cv2
from model import Deep3d
import numpy as np
from PIL import Image
from images2gif import writeGif


def inference(img_path):
    shape = (384, 160)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, shape)
    img_array = img.astype(np.float32)
    inputs = np.zeros((1, 160, 384, 3), dtype='float32')
    inputs[0] = img_array
    with tf.Graph().as_default():
        input = tf.placeholder(tf.float32, [None, 160, 384, 3], name='input_batch')
        net = Deep3d()
        right_img = net.build_model(input)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            in_dict = {input: inputs}
            right,mask = sess.run([right_img,net.mask], feed_dict=in_dict)
            right = np.clip(right[0].squeeze().transpose((0,1, 2)), 0, 255).astype(np.uint8)
            right = Image.fromarray(right,'RGB')
            right.save('output.jpg')
            print(">>>>>>>>" + str(mask.shape))
            depth_quatized = np.argmax(mask[0],axis=-1)

            depth_quatized = 32 - depth_quatized
            depth_quatized = depth_quatized*(255//33)
            depth_quatized = np.clip(depth_quatized, 0, 255).astype(np.uint8)

            depth_map = Image.fromarray(depth_quatized)
            depth_map.save('depth.jpg')

            depth_quatized = cv2.medianBlur(depth_quatized,5)
            depth_map = Image.fromarray(depth_quatized)
            depth_map.save('depth1.jpg')

            #left = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),'RGB')
            #cv2.imwrite('output.jpg',right)
            writeGif('demo.gif', [img, right], duration=0.08)


if __name__ == "__main__":
    inference(img_path='image.jpg')
