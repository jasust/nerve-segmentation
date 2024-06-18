import os
import cv2
import csv
import numpy as np
import tensorflow as tf
from itertools import chain
import matplotlib.pyplot as plt
from scipy.misc import imresize
from model import Model


def run_length_enc(label):
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

DEBUG = False
height = 420
width = 580

sess = tf.InteractiveSession()
model = Model(test=True)
model.load('records' + os.sep + 'testing_set.npy')
print('Dataset loaded...')

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('checkpoint' + os.sep + str(24))
saver.restore(sess, ckpt.model_checkpoint_path)
pred = []
for j in range(model.dataset_x.shape[0]//model.batch_size):
    pred.append(sess.run(model.pred, feed_dict={model.x: model.dataset_x[j*model.batch_size:(j+1)*model.batch_size]}))
pred = np.reshape(np.asarray(pred), (-1, 128, 128))

fpred = []
black = np.zeros((height, width))
kernel = np.ones((6, 6), np.float32)
for i in range(len(pred)):
    img = imresize(pred[i], (height, width)) > 0.5
    if np.sum(img) < 2800:
        fpred.append(black)
    else:
        fpred.append(cv2.dilate(np.array(img, np.float32), kernel))
fpred = np.array(fpred, np.float32)

if DEBUG:
    for i in range(fpred.shape[0]):
        plt.subplot(221)
        plt.imshow(np.reshape(model.dataset_x[i], (128, 128)), cmap="gray")
        plt.subplot(222)
        plt.imshow(pred[i], cmap="gray")
        plt.subplot(212)
        plt.imshow(fpred[i], cmap="gray")
        plt.show(block=True)

encpred = []
print("Writing submission file...")
for i in range(len(fpred)):
    encpred.append((i+1, run_length_enc(fpred[i])))
with open('records/submission16.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["img", "pixels"])
    for i in range(len(encpred)):
        writer.writerow([encpred[i][0], encpred[i][1]])
print("DONE")
