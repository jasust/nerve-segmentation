import os
import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imresize
from model import Model

DEBUG = False
height = 420
width = 580

def dice(y_true, y_pred):

    md = 0
    y_true_f = np.reshape(y_true, (y_true.shape[0], -1))
    y_pred_f = np.reshape(y_pred, (y_pred.shape[0], -1))
    for i in range(y_pred_f.shape[0]):
        union = np.sum(y_true_f[i]) + np.sum(y_pred_f[i])
        if union == 0:
            md += 1.0
        else:
            md += 2.*np.dot(y_true_f[i], y_pred_f[i])/union

    return md/y_pred_f.shape[0]

sess = tf.InteractiveSession()
model = Model()
model.load('records' + os.sep + 'validation_set.npy')
print('Dataset loaded...')

saver = tf.train.Saver()

if DEBUG:
    for i in range(16, 31, 2):
        ckpt = tf.train.get_checkpoint_state('checkpoint' + os.sep + str(i))
        print("Loading model... ", ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)

        start = time.time()
        acc = 0
        for j in range(model.dataset_x.shape[0]//model.batch_size):
            acc += sess.run(model.dice, feed_dict={model.x: model.dataset_x[j*model.batch_size:(j+1)*model.batch_size],
                                        model.y: model.dataset_y[j*model.batch_size:(j+1)*model.batch_size]})*100

        acc /= model.dataset_x.shape[0]//model.batch_size
        print("Validation Accuracy: %.2f%%" % acc)
        end = time.time()
        print("Average time per image: %.3fs" % ((end-start)/model.dataset_x.shape[0]))

ckpt = tf.train.get_checkpoint_state('checkpoint' + os.sep + str(24))
saver.restore(sess, ckpt.model_checkpoint_path)
pred = []
for j in range(model.dataset_x.shape[0]//model.batch_size):
    pred.append(sess.run(model.pred, feed_dict={model.x: model.dataset_x[j*model.batch_size:(j+1)*model.batch_size],
                                                model.y: model.dataset_y[j*model.batch_size:(j+1)*model.batch_size]}))
pred = np.reshape(np.asarray(pred), (-1, 128, 128))

print("Validation Accuracy on 128x128 images: %.2f%%" % (dice(model.dataset_y, pred)*100))

if DEBUG:
    for i in range(pred.shape[0]):
        plt.subplot(131)
        plt.imshow(np.reshape(model.dataset_x[i], (128, 128)), cmap="gray")
        plt.subplot(132)
        plt.imshow(pred[i], cmap="gray")
        plt.subplot(133)
        plt.imshow(model.dataset_y[i], cmap="gray")
        plt.show(block=True)

fpred = []
black = np.zeros((height, width))
for i in range(len(pred)):
    img = imresize(pred[i], (height, width)) > 0.5
    if np.sum(img) < 2800:
        fpred.append(black)
    else:
        fpred.append(img)
fpred = np.array(fpred, np.float32)

mask = []
original = []
data_path = 'validate'
for subdir, dirs, files in os.walk(data_path):
    for fileName in files:
        filePath = subdir + os.sep + fileName
        if fileName.split('.')[0].split('_')[-1] == 'mask':
            mask.append((int(fileName.split('_')[0])*1000+int(fileName.split('_')[1]), cv2.imread(filePath, 0)))
        else:
            original.append((int(fileName.split('_')[0])*1000+int(fileName.split('_')[1].split('.')[0]), cv2.imread(filePath, 1)))
mask.sort(key=lambda a: a[0])
original.sort(key=lambda a: a[0])

fmask = []
fori = []
for i in range(len(mask)):
    fmask.append(np.array(mask[i][1]).astype(np.float32)*(1./255))
    fori.append(np.array(original[i][1]))
fmask = np.array(fmask)
fori = np.array(fori)

if DEBUG:
    for i in range(10):
        plt.subplot(121)
        plt.imshow(fpred[i], cmap="gray")
        plt.subplot(122)
        plt.imshow(fmask[i], cmap="gray")
        plt.show(block=True)

print("Validation Accuracy on 420x580 images: %.2f%%" % (dice(fmask, fpred)*100))

spred = []
kernel = np.ones((6, 6), np.float32)
for i in range(len(fpred)):
    spred.append(cv2.dilate(fpred[i], kernel))
spred = np.array(spred)

if DEBUG:
    for i in range(fori.shape[0]):
        spred_plot = np.copy(fori[i][:][:][:])
        truth_plot = np.copy(fori[i][:][:][:])
        for j in range(height):
            for k in range(width):
                truth_plot[j][k][1] = truth_plot[j][k][1] * (1 - fmask[i][j][k])
                truth_plot[j][k][2] = truth_plot[j][k][2] * (1 - fmask[i][j][k])
                spred_plot[j][k][1] = spred_plot[j][k][1] * (1 - spred[i][j][k])
                spred_plot[j][k][2] = spred_plot[j][k][2] * (1 - spred[i][j][k])
        plt.subplot(121)
        plt.imshow(truth_plot)
        plt.subplot(122)
        plt.imshow(spred_plot)
        plt.show(block=True)

print("Validation Accuracy after postprocessing: %.2f%%" % (dice(fmask, spred)*100))
