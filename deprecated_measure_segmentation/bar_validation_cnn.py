import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
import cv2
import os


#Data importing, labeling and splitting:
def load_img(img_path, size = (1024, 1024), grayscale=True):
    """Load an image to the size of the parameter size."""
    return cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR), size)

def load_data_set(true_bars_path, false_bars_path):
    '''Loads the bar dataset from the two folders, one of correct bars and one of false bars'''
    files = os.listdir(true_bars_path)
    true_images = [load_img(filename) for filename in files]
    
    files = os.listdir(false_bars_path)
    false_images = [load_img(filename) for filename in files]
    
    labels = np.append(np.ones(len(true_images)), np.zeros(len(false_images)))
    images = true_images + false_images
    images = np.asarray(images)
    
    return images, labels
    
    
def split_train_test(images, labels, train_test_split = 0.9):
    ''' Splits the data set into x_train, y_train, x_test, y_test
    with the corresponding percentage of train_test_split'''
    n_images = len(labels)
    # Split at the given index
    split_index = int(train_test_split * n_images)
    shuffled_indices = np.random.permutation(n_images)
    train_indices = shuffled_indices[0:split_index]
    test_indices = shuffled_indices[split_index:]
    
    # Split the images and the labels
    x_train = images[train_indices, :, :]
    y_train = labels[train_indices]
    x_test = images[test_indices, :, :]
    y_test = labels[test_indices]
    
    return x_train, y_train, x_test, y_test


#The CNN:
def cnn(input_shape, optimizer = 'adam'):
    '''Returns a CNN Keras model to be run on the bar dataset'''
    # INPUTS
    # input_shape     - size of the input images
    # n_layers - number of layers
    # OUTPUTS
    # model    - compiled CNN

    # Define hyperparamters
    KERNEL = (3, 3)
    POOL_SIZE=(2, 2)
    

    # Define a model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=KERNEL,
                     activation='relu',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=POOL_SIZE))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # Print a summary of the model
    model.summary()

    return model

def run_cnn():
    #parameters
    image_size = (1024,1024)
    
    batch_size = 1024
    epochs = 12
    optimizer = 'adam'
    
    true_bars_path = r''
    false_bars_path = r''
    
    model_save_path = r''
    
    images, labels = load_data_set(true_bars_path, false_bars_path)
    x_train, y_train, x_test, y_test = split_train_test(images, labels, train_test_split = 0.9)
    cnn = cnn(image_size, optimizer = optimizer)
    
    cnn.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    cnn.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = cnn.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    cnn.save(model_save_path + r'accuracy_{0}_optimizer_{1}'.format(score[1], optimizer))