import os
import glob
from sklearn.model_selection import train_test_split
import shutil
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from numpy import array
from typing import Tuple
import cv2
import tensorflow as tf

def split_data(data_train_path: str, save_train_path: str, save_val_path: str, img_extension: str, split_size=0.2)\
        -> None:
    """

    :param data_train_path:
    :param save_train_path:
    :param save_val_path:
    :param img_extension:
    :param split_size: the proportion of data you wish to use for validation, by default 20%.
    :return:
    """

    folders = os.listdir(data_train_path)

    for folder in folders:

        full_path = os.path.join(data_train_path, folder)
        images_path = glob.glob(os.path.join(full_path, img_extension))

        x_train, x_val = train_test_split(images_path, test_size=split_size)

        for x in x_train:

            folder_path = os.path.join(save_train_path, folder)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            shutil.copy(x, folder_path)

        for x in x_val:

            folder_path = os.path.join(save_val_path, folder)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            shutil.copy(x, folder_path)


def create_data_generator(batch_size: int, data_train_path: str, data_val_path: str, data_test_path: str, height: int,
                          width: int):
    """
    :param batch_size: we split the sample into a defined number of samples that will pass through the neural
     network at the same time
    :param data_train_path: Path to the folders containing the training data
    :param data_val_path: Path to the folders containing the test data
    :param data_test_path: Path to the files containing the validation data
    :param height: height we want for our images, all our images will be resized
    :param width: width we want for our images, all our images will be resized
    :return: We obtain 3 data sets in the form of a keras object
    """

    # It is assumed that for training, test and validation data the images have the same dimensions

    train_preprocessor = ImageDataGenerator(
        rescale=1 / 255.,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    test_preprocessor = ImageDataGenerator(
        rescale=1 / 255.
    )

    train_generator = train_preprocessor.flow_from_directory(
        data_train_path,
        class_mode="categorical",
        target_size=(height, width),
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size
    )

    val_generator = test_preprocessor.flow_from_directory(
        data_val_path,
        class_mode="categorical",
        target_size=(height, width),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        data_test_path,
        class_mode="categorical",
        target_size=(height, width),
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    return train_generator, val_generator, test_generator

def get_data_train(data_train_path: str, image_extension: str, width: int, height: int) -> Tuple[list, array]:
    """
    :param data_train_path: Path to the folders containing the training data
    :param image_extension: extension of the images to be imported. Be careful, the images must have the same extension
    :param height: height we want for our images, all our images will be resized
    :param width: width we want for our images, all our images will be resized
    :return: a list with labels for each images and a 4 dimensional numpy array with our images
    """

    data_train = ()
    index_train_list = []
    x_data_train = []

    for file in glob.glob(data_train_path):
        image_label = os.listdir(file)
        data_train = []
        index_train_list = []

        for i in range(len(image_label)):
            full_path = os.path.join(data_train_path, image_label[i], image_extension)
            dtrain = [cv2.imread(file) for file in glob.glob(os.path.join(full_path))]
            dtrain_resized = []

            for j in range(len(dtrain)):
                img_resized = cv2.resize(dtrain[j], (width, height), interpolation=cv2.INTER_AREA)
                dtrain_resized.append(img_resized)

            dtrain = np.array(dtrain_resized)
            data_train.append(dtrain)

            for k in range(dtrain.shape[0]):
                index_train_list.append(image_label[i])

            print(f'Number of train pictures in the folder "{image_label[i]}": {len(dtrain)}')
    print(f'\n'
          f'\nindex_train_list length : {len(index_train_list)}\n')

    x_data_train = np.concatenate(data_train, axis=0)
    print(f'Shape of x_data_train tensor : {x_data_train.shape}\n')


    return index_train_list, x_data_train

def get_data_test(data_test_path: str, image_extension: str, width: int, height: int) -> Tuple[list, array]:
    """
    :param data_test_path: Path to the folder containing the images for the tests
    :param image_extension: extension of the images to be imported. Be careful, the images must have the same extension
    :param height: height we want for our images, all our images will be resized
    :param width: width we want for our images, all our images will be resized
    :return: a list with labels for each images and a 4 dimensional numpy array with our images
    """

    data_test = ()
    index_test_list = []
    x_data_test = []

    for file in glob.glob(data_test_path):
        image_label = os.listdir(file)
        data_test = []
        index_test_list = []

        for i in range(len(image_label)):
            full_path = os.path.join(data_test_path, image_label[i], image_extension)
            dtest = [cv2.imread(file) for file in glob.glob(os.path.join(full_path))]
            dtest_resized = []

            for j in range(len(dtest)):
                img_resized = cv2.resize(dtest[j], (width, height), interpolation=cv2.INTER_AREA)
                dtest_resized.append(img_resized)

            dtest = np.array(dtest_resized)
            data_test.append(dtest)

            for k in range(dtest.shape[0]):
                index_test_list.append(image_label[i])

            print(f'Number of test pictures in the folder "{image_label[i]}": {len(dtest)}')
        print(f'\nindex_train_list length : {len(index_test_list)}\n')

        x_data_test = np.concatenate(data_test, axis=0)
        print(f'Shape of x_data_test : {x_data_test.shape}\n')


    return index_test_list, x_data_test


def get_data_val(data_val_path: str, image_extension: str, width: int, height: int) -> Tuple[list, array]:
    """
    :param data_vel_path: Path to the folder containing the images for validation
    :param image_extension: extension of the images to be imported. Be careful, the images must have the same extension
    :param height: height we want for our images, all our images will be resized
    :param width: width we want for our images, all our images will be resized
    :return: a list with labels for each images and a 4 dimensional numpy array with our images
    """

    data_val = ()
    index_val_list = []
    x_data_val = []

    for file in glob.glob(data_val_path):
        image_label = os.listdir(file)
        data_val = []
        index_val_list = []

        for i in range(len(image_label)):
            full_path = os.path.join(data_val_path, image_label[i], image_extension)
            dval = [cv2.imread(file) for file in glob.glob(os.path.join(full_path))]
            dval_resized = []

            for j in range(len(dval)):
                img_resized = cv2.resize(dval[j], (width, height), interpolation=cv2.INTER_AREA)
                dval_resized.append(img_resized)

            dval = np.array(dval_resized)
            data_val.append(dval)

            for k in range(dval.shape[0]):
                index_val_list.append(image_label[i])

            print(f'Number of val pictures in the folder "{image_label[i]}": {len(dval)}')
    print(f'\nindex_train_list length : {len(index_val_list)}\n')

    x_data_val = np.concatenate(data_val, axis=0)
    print(f'Shape of x_data_val: {x_data_val.shape}')


    return index_val_list, x_data_val

def get_label_dictionary(data_path: str) -> dict:
    """
    :param data_path: Path to the folders containing the training data
    :return: A symmetrical dictionary containing the labels and their numerical values
    """

    label_list = os.listdir(data_path)
    label_dictionary = {}

    for i in range(len(label_list)):
        label_dictionary[label_list[i]] = i

    for i in range(len(label_list)):
        label_dictionary[i] = label_list[i]

    return label_dictionary


def numerical_transformation(y_train: list, y_test: list, y_val: list, data_path: str) -> Tuple[array, array, array]:
    """
    :param y_train: list of raw labels -in string form- of training data
    :param y_test: list of raw labels -in string form- of the test data
    :param y_val: list of raw labels -in string form- of validation data
    :param data_path: Path to the folders containing the training data
    :return:
    """

    label_dictionary = get_label_dictionary(data_path)
    y_test_numerical = []
    y_train_numerical = []
    y_val_numerical = []

    for i in range(len(y_train)):
        y_train_numerical.append(label_dictionary[y_train[i]])
    print(f'\nLength of numerical train labels: {len(y_train_numerical)}\nLength of train labels: {len(y_train)}\n')

    for i in range(len(y_test)):
        y_test_numerical.append(label_dictionary[y_test[i]])
    print(f'Lenght of numerical test labels: {len(y_test_numerical)}\nLength of test labels: {len(y_test)}\n')

    for i in range(len(y_val)):
        y_val_numerical.append(label_dictionary[y_val[i]])
    print(f'Length of numerical validation labels: {len(y_val_numerical)}\nLength of validation labels: {len(y_val)}\n')

    y_train_numerical = np.array(y_train_numerical)
    y_test_numerical = np.array(y_test_numerical)
    y_val_numerical = np.array(y_val_numerical)

    return y_test_numerical, y_train_numerical, y_val_numerical


def data_preparation(train_path: str, test_path: str, val_path: str, extension: str, width: int, height: int) -> \
        Tuple[array, array, array, array, array, array]:

    raw_y_train, x_train = get_data_train(train_path, extension, width, height)
    raw_y_test, x_test = get_data_test(test_path, extension, width, height)
    raw_y_val, x_val = get_data_val(val_path, extension, width, height)

    y_test, y_train, y_val = numerical_transformation(raw_y_train, raw_y_test, raw_y_val, train_path)

    # We normalise our data to allow the neural network to work properly
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_val = x_val.astype('float32') / 255

    return x_train, x_test, y_test, y_train, x_val, y_val