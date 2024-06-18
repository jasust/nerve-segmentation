import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False
height = 128
width = 128

def resize(file_path):
    img = cv2.imread(file_path, 0)
    img = cv2.resize(img, (height,width))
    return img

data_path = input("Enter path to target folder: ")
name = input("Enter name of dataset: ")

x = []
y = []
start = time.time()
print('Loading data...')
for subdir, dirs, files in os.walk(data_path):
    for fileName in files:
        filePath = subdir + os.sep + fileName
        if fileName.split('.')[0].split('_')[-1] == 'mask':
            y.append((int(fileName.split('_')[0])*1000 + int(fileName.split('_')[1]), resize(filePath)))
        else:
            if '_' in fileName:
                x.append((int(fileName.split('_')[0])*1000+int(fileName.split('_')[1].split('.')[0]), resize(filePath)))
            else:
                x.append((int(fileName.split('.')[0]), resize(filePath)))

x.sort(key=lambda a: a[0])
y.sort(key=lambda a: a[0])

if DEBUG:
    for i in range(5):
        plt.subplot(121)
        plt.imshow(x[i][1], cmap="gray")
        plt.subplot(122)
        plt.imshow(y[i][1], cmap="gray")
        plt.show(block=True)

dataset = []
num_examples = len(x)
print('Writing data...')
filename = 'records' + os.sep + name
for i in range(num_examples):
    dataset.append(np.array(x[i][1]))
    if len(y) > 0:
        dataset.append(np.array(y[i][1]))
np.save(filename, np.array(dataset))

end = time.time() - start
print("Number of examples : %.4r" % num_examples)
print("Time: %.2f seconds" % end)
