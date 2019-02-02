#!/home/pi/Desktop/virtualenvs/PD/bin/python3
import cv2
import numpy as np
import os         
from random import shuffle 
from tqdm import tqdm      
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

#%matplotlib inline


TRAIN_DIR = 'train'
TEST_DIR = 'test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'plants-disease-convnet'

def create_label(image_name):
    word_label = image_name[0]
    if word_label == 'h':
        return [1,0,0,0,0,0,0,0,0]
    if word_label == 'a':
        return [0,1,0,0,0,0,0,0,0]
    elif word_label == 'e':
        return [0,0,1,0,0,0,0,0,0]
    elif word_label == 'l':
        return [0,0,0,1,0,0,0,0,0]
    elif word_label == 'r':
        return [0,0,0,0,1,0,0,0,0]
    elif word_label == 's':
        return [0,0,0,0,0,1,0,0,0]
    elif word_label == 'm':
        return [0,0,0,0,0,0,1,0,0]
    elif word_label == 'p':
        return [0,0,0,0,0,0,0,1,0]
    elif word_label == 'o':
        return [0,0,0,0,0,0,0,0,1]
    

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = create_label(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data



train_data = create_train_data()




tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE,3], name='input')

convnet = conv_2d(convnet, 32, 10, activation='relu')
convnet = max_pool_2d(convnet, 10)

convnet = conv_2d(convnet,64, 10,activation='relu')
convnet = max_pool_2d(convnet, 10)

convnet = conv_2d(convnet, 128, 10, activation='relu')
convnet = max_pool_2d(convnet, 10)

convnet = conv_2d(convnet, 32, 10, activation='relu')
convnet = max_pool_2d(convnet, 10)

convnet = conv_2d(convnet, 64, 10, activation='relu')
convnet = max_pool_2d(convnet, 10)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 9, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

##model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10,validation_set({'input':X_test}, {'targets': y_test}),
##          snapshot_step=500, show_metric=True,run_id=MODEL_NAME)

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model_loaded')

train = train_data[:-100]
test = train_data[-100:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE, IMG_SIZE,3)
test_y = [i[1] for i in test]


model.fit({'input': X}, {'targets': Y}, n_epoch=20, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)











