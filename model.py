import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import slim


class Model:

    def __init__(self, test=False, training=False):

        self.DEBUG = False
        self.test = test
        self.training = training

        self.height = 128
        self.width = 128

        self.num_epochs = 30
        self.batch_size = 12
        self.display_step = 94
        self.point_step = 47
        self.save_step = 2

        self.loss_f = []
        self.acc_f = []

        def u_net(data, is_training):

            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME', normalizer_fn=slim.batch_norm,
                                normalizer_params={'decay': 0.9997, 'is_training': True, 'updates_collections': None,
                                                   'trainable': is_training},
                                weights_initializer=slim.initializers.xavier_initializer(),
                                weights_regularizer=slim.l2_regularizer(0.0005)):

                conv1 = slim.repeat(data, 2, slim.conv2d, 32, [3, 3], scope='conv1')
                pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')
                conv2 = slim.repeat(pool1, 2, slim.conv2d, 64, [3, 3], scope='conv2')
                pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')
                conv3 = slim.repeat(pool2, 2, slim.conv2d, 128, [3, 3], scope='conv3')
                pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')
                conv4 = slim.repeat(pool3, 2, slim.conv2d, 256, [3, 3], scope='conv4')
                pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')

                conv5 = slim.repeat(pool4, 2, slim.conv2d, 512, [3, 3], scope='conv5')
                deconv1 = slim.conv2d_transpose(conv5, 512, stride=2, kernel_size=2)
                concat1 = tf.concat([conv4, deconv1], 3, name='concat1')
                conv6 = slim.repeat(concat1, 2, slim.conv2d, 256, [3, 3], scope='conv6')
                deconv2 = slim.conv2d_transpose(conv6, 256, stride=2, kernel_size=2)
                concat2 = tf.concat([conv3, deconv2], 3, name='concat2')
                conv7 = slim.repeat(concat2, 2, slim.conv2d, 128, [3, 3], scope='conv7')
                deconv3 = slim.conv2d_transpose(conv7, 128, stride=2, kernel_size=2)
                concat3 = tf.concat([conv2, deconv3], 3, name='concat3')
                conv8 = slim.repeat(concat3, 2, slim.conv2d, 64, [3, 3], scope='conv8')
                deconv4 = slim.conv2d_transpose(conv8, 64, stride=2, kernel_size=2)
                concat4 = tf.concat([conv1, deconv4], 3, name='concat4')

                conv9 = slim.repeat(concat4, 2, slim.conv2d, 32, [3, 3], scope='conv9')
                conv1x1 = slim.conv2d(conv9, 2, [1, 1], activation_fn=tf.nn.sigmoid, scope='conv1x1')

            return conv1x1

        def dice_coef(y_true, y_pred):

            md = tf.constant(0.0)
            y_true = tf.cast(y_true, tf.float32)
            y_true_f = slim.flatten(y_true)
            y_pred_f = slim.flatten(y_pred)
            for i in range(self.batch_size):
                union = tf.reduce_sum(y_true_f[i]) + tf.reduce_sum(y_pred_f[i])
                md = tf.cond(tf.equal(union, 0.0), lambda: tf.add(md, 1.0),
                             lambda: tf.add(md, tf.div(2.*tf.reduce_sum(tf.multiply(y_true_f[i], y_pred_f[i])), union)))

            return tf.div(md, self.batch_size)

        self.x = tf.placeholder("float", [None, self.height, self.width, 1])
        self.y = tf.placeholder("int32", [None, self.height, self.width])

        logits = u_net(self.x, is_training=self.training)

        pred = tf.argmax(logits, dimension=3)
        pred = tf.reshape(pred, [-1, self.height, self.width, 1])
        pred = tf.cast(pred, tf.float32)
        self.pred = pred

        class_weight = tf.constant([0.75, 0.25])
        weighted_logits = tf.multiply(logits, class_weight)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=weighted_logits, labels=self.y)

        self.loss = tf.reduce_mean(cross_entropy)
        self.dice = dice_coef(self.y, pred)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005, use_locking=False).minimize(self.loss)

        print("Initialization finished...")

    def load(self, file_name):

        data = np.load(file_name)

        if self.test:
            image = data
        else:
            image = data[::2][:][:]
            mask = data[1::2][:][:]

        image = image.astype(np.float32)*(1./255)-0.5
        if not self.test:
            mask = (mask.astype(np.float32)*(1./255)).astype(np.int32)

        if self.training:
            image = np.vstack((image, np.flip(image, 2)))
            if not self.test:
                mask = np.vstack((mask, np.flip(mask, 2)))

        self.dataset_x = np.reshape(image, (-1, self.height, self.width, 1))
        if not self.test:
            self.dataset_y = np.reshape(mask, (-1, self.height, self.width))

        if self.DEBUG:
            plt.subplot(221)
            plt.imshow(image[0], cmap="gray")
            plt.subplot(222)
            plt.imshow(mask[0], cmap="gray")
            plt.subplot(223)
            plt.imshow(image[839], cmap="gray")
            plt.subplot(224)
            plt.imshow(mask[839], cmap="gray")
            plt.show(block=True)

    def train(self):

        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for i in range(self.num_epochs):

                step = 1
                start_time = time.time()

                while step * self.batch_size <= len(self.dataset_x):

                    batch_x = self.dataset_x[(step - 1) * self.batch_size:step * self.batch_size]
                    batch_y = self.dataset_y[(step - 1) * self.batch_size:step * self.batch_size]

                    sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y})

                    if step % self.point_step == 0:

                        loss_value, dice_value = sess.run([self.loss, self.dice],
                                                          feed_dict={self.x: batch_x, self.y: batch_y})
                        self.loss_f.append(loss_value)
                        self.acc_f.append(100 * dice_value)

                        if step % self.display_step == 0:
                            print("Epoch %d, step %d: loss = %f, dice = %.4f" % (i+1, step, loss_value, dice_value))

                    step += 1

                duration = time.time() - start_time
                print("Duration of training for this epoch: %.2fs" % duration)
                if (i+1) % self.save_step == 0:
                    saver.save(sess, 'checkpoint' + os.sep + 'model.ckpt', i+1)

            print("Optimization Finished...")

        plt.plot(self.acc_f)
        plt.title("Accuracy in %")
        plt.savefig('records\\acc.png')
        plt.close()
        plt.plot(self.loss_f)
        plt.title("Loss function")
        plt.savefig('records\\loss.png')
        plt.close()

if __name__ == '__main__':

    model = Model(training=True)
    model.load('records' + os.sep + 'training_set.npy')
    model.train()
