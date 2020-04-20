import os
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import neuralnet
from tensorflow.keras import datasets, layers, models, optimizers, callbacks
import changelr


WIDTH = 120
HEIGHT = 100
OUTPUT = 10
OUTPUT_DIM = 4

def load_captchas():
    path = './data4len4digit'
    dirs = ['train', 'test']
    #alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    alphabet = '0123456789'

    train_c = []
    train_l = []
    test_c = []
    test_l = []
    captchas = [train_c, test_c]
    labels = [train_l, test_l]
    for i in range(len(dirs)):
        folder = dirs[i]
        q = 0
        for fn in os.listdir(os.path.join(path, folder)):
            if fn.endswith('.png'):
                q+=1
                fpath = os.path.join(path, folder, fn)
                im = np.asarray(Image.open(fpath).convert('L').resize((WIDTH, HEIGHT), Image.ANTIALIAS))
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
            if(q % 1000 == 0):
                print(q)


    for i in range(2):
        captchas[i] = np.array(captchas[i])
        labels[i] = np.array(labels[i])

    return (captchas, labels)

def formatData():
    trainImages = captchas[0]
    testImages = captchas[1]

    trainImages = trainImages.reshape(-1,WIDTH,HEIGHT,1).astype('float32') / 255.0
    testImages = testImages.reshape(-1,WIDTH,HEIGHT,1).astype('float32') / 255.0

    trainLabels = labels[0]
    testLabels = labels[1]
    trainLabels = tf.reshape(trainLabels, [-1, OUTPUT_DIM, OUTPUT])
    testLabels = tf.reshape(testLabels, [-1, OUTPUT_DIM, OUTPUT])

    #print(trainLabels.shape)

    return (trainImages, testImages, trainLabels, testLabels)


def getMostUncertainSamples(predictions, correctIndex, currentTestImages, currentTestLabels):

    uncer = []
    # Calculate uncertainties according to formula in paper
    for k in correctIndex:
        predUncer = 0
        for i in range(OUTPUT_DIM):
            normalized = predictions[k][i] / sum(predictions[k][i])
            normMax = np.amax(normalized)
            maxDiv = (np.amax(normalized / normMax))/normMax
            predUncer += maxDiv

        predUncer = predUncer / OUTPUT_DIM
        uncer.append(predUncer)

    toConcat = []
    toConcatLabels = []
    numAdded = 0

    if(len(uncer) > 0):
        perc = np.percentile(uncer, 90)
        # Add uncertain samples to training data 
        for i, val in enumerate(uncer):
            if(val >= perc):
                numAdded+=1
                indexImage = int(correctIndex[i])
                toConcat.append(currentTestImages[indexImage,:,:])
                toConcatLabels.append(currentTestLabels[indexImage,:,:])

    return toConcat, toConcatLabels, numAdded


def ADL(model, trainImages, trainLabels, testImages, testLabels):
    currentTrainImages = trainImages
    currentTrainLabels = trainLabels

    currentTestImages = testImages
    currentTestLabels = testLabels

    # Callback to change learning rate of SGD
    ChangeLearningRate = changelr.ChangeLearningRate()
    
    # Each i is one ADL epoch
    for i in range(5):
        print('Re-training iteration:', i)

        history = model.fit(x=currentTrainImages, y=currentTrainLabels,
                               validation_data = (currentTestImages, currentTestLabels), epochs = 10, batch_size = 64, callbacks=[ChangeLearningRate])

        predictions = model.predict(currentTestImages)

        correctIndex = []
        wrong = []

        for i in range(len(predictions)):
            isEqual = True
            for j in range(len(currentTestLabels[0])):
                currLabel = currentTestLabels[i][j].numpy().argmax()
                currPredict = predictions[i][j].argmax()
                if(currLabel != currPredict):
                    isEqual = False
                    break 
            if(isEqual):
                correctIndex.append(i)
            else:
                wrong.append(i)

        # print(len(correctIndex))
        # print(len(wrong))

        toConcat, toConcatLabels, numAdded = getMostUncertainSamples(predictions, correctIndex, currentTestImages, currentTestLabels)

        if(len(toConcat) > 0):
            currentTrainImages = np.concatenate([currentTrainImages, toConcat])
            currentTrainLabels = np.concatenate([currentTrainLabels, toConcatLabels])
        #print("Added", numAdded)

    return history




captchas, labels = load_captchas()


trainImages, testImages, trainLabels, testLabels = formatData()


nn = neuralnet.NeuralNet(100, 10000)
model = nn.generateModel()
model.summary()

sgd = optimizers.SGD(lr = 0.01, momentum = 0.9)
model.compile(sgd, loss = 'mean_squared_error', metrics=['accuracy'])

history = ADL(model, trainImages, trainLabels, testImages, testLabels)

