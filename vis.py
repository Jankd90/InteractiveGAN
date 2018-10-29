
# -*- coding: utf-8 -*-

# generate new kinds of pokemons

import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *

slim = tf.contrib.slim

HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 5000
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
version = 'newPokemon'
newPoke_path = './' + version

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
colors = [(255,0,0),(0,0,255),(0,255,0),(255,255,255)]
pts = [(0,0),(0,0),(0,0),(0,0)]
pointIndex = 0

cam = cv2.VideoCapture(0)

_,img = cam.read()
_,img = cam.read()

# Aspect ratio for an A4 sheet. 1:1.414
# 500 * 1.414 = 707, that is why I chose this size.
ASPECT_RATIO = (500,707)

pts2 = np.float32([[0,0],[ASPECT_RATIO[1],0],[0,ASPECT_RATIO[0]],[ASPECT_RATIO[1],ASPECT_RATIO[0]]])
# mouse callback function
def draw_circle(event,x,y,flags,param):
	global img
	global pointIndex
	global pts

	if event == cv2.EVENT_LBUTTONDOWN:
		if(pointIndex<4):
		    print(pointIndex)
		    cv2.circle(img,(x,y),10,colors[pointIndex],-1)
		    pts[pointIndex] = (x,y)
		    pointIndex = pointIndex + 1

def selectFourPoints():
	global img
	global pointIndex

	print ("Please select 4 points, by double clicking on each of them in the order: \n\
	top left, top right, bottom left, bottom right.")


	while(pointIndex != 4):
		cv2.imshow('image',img)
		key = cv2.waitKey(20) & 0xFF
		if key == 27:
			return False


	return True
# Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)


def getSubAverages(img, dims=100):
    subAverages = np.zeros(dims)

    sizex, sizey = map(lambda num: int(num / math.sqrt(dims)), img.shape)

    for i in range(10):

        for j in range(10):
            x = i * sizex

            y = j * sizey

            index = j + (i * 10)

            sub = img[x:x + sizex, y:y + sizey]

            subAverages[index] = (np.sum(sub) / (sizex * sizey))

    return np.interp(subAverages, (40, 140), (-1, 1)).reshape(1,dims)

def lrelu(x, n, leak=0.2):
    return tf.maximum(x, leak * x, name=n)

def process_data():
    current_dir = '/home/jan/Documents/data/POKEMON'
    # parent = os.path.dirname(current_dir)
    pokemon_dir = os.path.join(current_dir)
    images = []
    for each in os.listdir(pokemon_dir):
        images.append(os.path.join(pokemon_dir,each))
    # print images
    all_images = tf.convert_to_tensor(images, dtype = tf.string)

    images_queue = tf.train.slice_input_producer(
                                        [all_images])

    content = tf.read_file(images_queue[0])
    image = tf.image.decode_jpeg(content, channels = CHANNEL)
    # sess1 = tf.Session()
    # print sess1.run(image)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
    # noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise'))
    # print image.get_shape()
    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])
    # image = image + noise
    # image = tf.transpose(image, perm=[2, 0, 1])
    # print image.get_shape()

    image = tf.cast(image, tf.float32)
    image = image / 255.0

    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)

    return iamges_batch, num_images

def generator(input, random_dim, is_train, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 32 # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gen') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
         #Convolution, bias, activation, repeat!
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        #Convolution, bias, activation, repeat!
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')

        #128*128*3
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6


def discriminator(input, is_train, reuse=False):
    c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Convolution, activation, bias, repeat!
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(conv1, n='act1')
         #Convolution, activation, bias, repeat!
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        #Convolution, activation, bias, repeat!
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
         #Convolution, activation, bias, repeat!
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')

        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')


        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

        # wgan just get rid of the sigmoid
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        acted_out = tf.nn.sigmoid(logits)
        return logits #, acted_out


def train():
    random_dim = 100
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    with tf.variable_scope('input'):
        #real and fake image placholders
        real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        is_train = tf.placeholder(tf.bool, name='is_train')

    # wgan
    fake_image = generator(random_input, random_dim, is_train)

    real_result = discriminator(real_image, is_train)
    fake_result = discriminator(fake_image, is_train, reuse=True)

    d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(real_result)  # This optimizes the discriminator.
    g_loss = -tf.reduce_mean(fake_result)  # This optimizes the generator.


    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gen' in var.name]
    # test
    # print(d_vars)
    trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]


    batch_size = BATCH_SIZE
    image_batch, samples_num = process_data()

    batch_num = int(samples_num / batch_size)
    total_batch = 0
    sess = tf.Session()
    saver = tf.train.Saver()
    
    # continue training
    #save_path = saver.save(sess, "./model/"+version+"/model.ckpt")
    #ckpt = tf.train.latest_checkpoint('./model/' + version)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    saver.restore(sess, './model/' +version + '/1750')
    print("Model restored.")

    #saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    print('start vis...')

    while (True):
        if (selectFourPoints()):
            pts1 = np.float32([ \
                [pts[0][0], pts[0][1]], \
                [pts[1][0], pts[1][1]], \
                [pts[2][0], pts[2][1]], \
                [pts[3][0], pts[3][1]]])

            M = cv2.getPerspectiveTransform(pts1, pts2)
            while (True):
                _, frame = cam.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dst = cv2.warpPerspective(frame, M, (707, 500))
                #sample_noise = np.random.uniform(-1.0, 1.0, size=[1, random_dim]).astype(np.float32)
                sample_noise = getSubAverages(dst).astype(np.float32)
                imgtest = sess.run(fake_image, feed_dict={random_input: sample_noise, is_train: False})

                cv2.imshow("output", dst)
        # Capture frame-by-frame
        #print(imgtest.shape)
        #gray = cv2.cvtColor(imgtest, cv2.COLOR_BGR2GRAY)
        # Our operations on the frame come here
                color = cv2.cvtColor(imgtest[0], cv2.COLOR_BGR2RGB)
                big = cv2.resize(color, (0, 0), fx=4, fy=4)
        # Display the resulting frame
                cv2.imshow('window', big)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyWindow('image')
                #break
                if cv2.waitKey(1) & 0xFF == ord('w'):
                    break



if __name__ == "__main__":
    train()
    # test()
