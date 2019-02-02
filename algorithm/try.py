def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpictures'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME ='plants-disease-convnet'
    

    
    print("hey")
    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 10, activation='relu')
    convnet = max_pool_2d(convnet, 10)

    convnet = conv_2d(convnet, 64, 10, activation='relu')
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

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]
        print(len(img_data))
        print(len(img_data[0]))
        print(len(img_num))

        y = fig.add_subplot(4, 4, num + 1)
        orig = img_data
        
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)

        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            str_label = 'healthy'
        elif np.argmax(model_out) == 1:
            str_label='antracnose'
        elif np.argmax(model_out) == 2:
            str_label = 'early blight'
        elif np.argmax(model_out) == 3:
            str_label = 'lateblight'
        elif np.argmax(model_out) == 4:
            str_label = 'rust'
        elif np.argmax(model_out) == 5:
            str_label = 'spot'
        elif np.argmax(model_out) == 6:
            str_label = 'mosaic'
        elif np.argmax(model_out) == 7:
            str_label = 'powdery'
        elif np.argmax(model_out) == 8:
            str_label = 'object'

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    plt.show()


    

##    for num, data in enumerate(verify_data):
##
##        img_num = data[1]
##        img_data = data[0]
##
##        y = fig.add_subplot(3, 4, num + 1)
##        orig = img_data
##        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
##        # model_out = model.predict([data])[0]
##        model_out = model.predict([data])[0]
##
##        if np.argmax(model_out) == 0:
##            str_label = 'healthy'
##        elif np.argmax(model_out) == 1:
##            str_label = 'bacterial'
##        elif np.argmax(model_out) == 2:
##            str_label = 'viral'
##        elif np.argmax(model_out) == 3:
##            str_label = 'lateblight'
##
##        if str_label =='healthy':
##            status ="HEALTHY"
##            print(status)
##        else:
##            status = "UNHEALTHY"
##            print(status)
##
##        if str_label == 'bacterial':
##            diseasename = "Bacterial Spot "
##            print("Disease:"+ diseasename)
##
##        elif str_label == 'viral':
##            diseasename = "Yellow leaf curl virus "
##            
##        elif str_label == 'lateblight':
##            diseasename = "Late Blight "
##            print("Disease:"+ diseasename)


analysis()
