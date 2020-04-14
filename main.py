import os
import argparse
import datetime
import sys
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import neuralnet
from tensorflow.keras import datasets, layers, models, optimizers


def load_captchas():
    path = './data'
    dirs = ['train', 'test']
    alphabet = '0123456789'

    captcha_info = os.path.join(path, 'info.json')
    with open(captcha_info, 'r') as f:
        info = json.load(f)

    train_c = []
    train_l = []
    test_c = []
    test_l = []
    captchas = [train_c, test_c]
    labels = [train_l, test_l]
    for i in range(len(dirs)):
        folder = dirs[i]
        for fn in os.listdir(os.path.join(path, folder)):
            if fn.endswith('.png'):
                fpath = os.path.join(path, folder, fn)
                im = np.asarray(Image.open(fpath).convert('L').resize((120, 100), Image.ANTIALIAS))
                #im = im.reshape(120*100)
                captchas[i].append(im)
                data = os.path.basename(fpath).split('_')[0]
                label = []
                for d in data:
                    ind = alphabet.index(d)
                    da = [0] * len(alphabet)
                    da[ind] = 1
                    label.extend(da)
                labels[i].append(label)


    for i in range(2):
        captchas[i] = np.array(captchas[i])
        labels[i] = np.array(labels[i])

    print(info)
    return (info, captchas, labels)

## For testing if network works
# def formatDataMNIST():
#     (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#     train_images = train_images.reshape(60000, 28, 28, 1).astype('float32') / 255.0
#     test_images = test_images.reshape(10000, 28, 28, 1).astype('float32') / 255.0

#     train_labels = tf.keras.utils.to_categorical(train_labels, 10)
#     test_labels = tf.keras.utils.to_categorical(test_labels, 10)
#     return (train_images, train_labels, test_images, test_labels)

#trainImages, trainLabels, testImages, testLabels = formatDataMNIST()

def formatData():
    trainImages = captchas[0]
    testImages = captchas[1]
    # plt.imshow(trainImages[73])
    # plt.show()

    trainImages = trainImages.reshape(-1,120,100,1).astype('float32') / 255.0
    testImages = testImages.reshape(-1,120,100,1).astype('float32') / 255.0

    trainLabels = labels[0]
    testLabels = labels[1]
    trainLabels = tf.reshape(trainLabels, [-1, 4, 10])
    testLabels = tf.reshape(testLabels, [-1, 4, 10])


    return (trainImages, testImages, trainLabels, testLabels)



info, captchas, labels = load_captchas()
trainImages, testImages, trainLabels, testLabels = formatData()

nn = neuralnet.NeuralNet(100, 10000)
model = nn.generateModel()
model.summary()

adam = optimizers.Adam(lr = 0.0001)
model.compile(adam, loss = 'mean_squared_error', metrics=['accuracy'])
history = model.fit(x=trainImages, y=trainLabels, epochs = 150, validation_data=(testImages, testLabels), batch_size = 64)

score = model.evaluate(testImages, testLabels, batch_size = 64)


