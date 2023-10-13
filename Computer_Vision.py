import tensorflow as tf
import numpy as np
from keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import classification_report, confusion_matrix
from numpy import array

from utilities import create_data_generator, data_preparation, get_label_dictionary


def parse_arguments():

    parser = argparse.ArgumentParser(description='Choose parameters and hyperparameters for data and models')

    parser.add_argument("-e", "--data_train_path", type=str, required=False, metavar='',
                        default="./Data/train_set/train/")
    parser.add_argument("-t", "--data_test_path", type=str, required=False, metavar='',
                        default="./Data/seg_test/")
    parser.add_argument("-p", "--data_pred_path", type=str, required=False, metavar='',
                        default="./Data/seg_pred")
    parser.add_argument("-j", "--data_val_path", type=str, required=False, metavar='',
                        default="./Data/train_set/validation/")
    parser.add_argument("-i", "--image_extension", type=str, required=False, metavar='', help=" | [jpg, png, jpeg, tif]",
                        default="*.jpg", choices=["*.jpg", "*.png", "*.jpeg", "*.tif"])
    parser.add_argument("-m", "--metrics", type=str, required=False, default="accuracy", metavar='',
                        help=" | [accuracy, recall, precision]", choices=["accuracy", "recall", "precision"])
    parser.add_argument("-b", "--batch_size", help=' | Choose multiple of 2 preferably: [2, 4, 8, 16, 32, 64, 128]. Value'
                                                   'by default : 32', type=int, required=False, default=32, metavar='',
                        choices=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("-n", "--epochs_number", type=int, required=False, metavar='', default=5)
    parser.add_argument("-f", "--filters", help=" | Number of filters for convolution, this number is"
                                                     " multiplied by 2 for each stack of layers:"
                                                     "[2, 4, 8, 16, 32, 64, 128].  Value by default :2 ",
                        type=int, required=False, default=2, choices=[2, 4, 8, 16, 32, 64, 128], metavar='')
    parser.add_argument("-plot", "--plot", help=' | Do you want to plot 25 random images from the train folder? Default'
                                                ' on True, write -plot to make it false', action='store_false')
    parser.add_argument("-evaluate", "--evaluation", help=" | Do you want to evaluate the latest trained model"
                                                            " on validation and test data? Default on False,"
                                                            " write -evaluation to make it true", action='store_true')
    parser.add_argument("-training", "--training", action="store_true",
                        help=' | Train the model, you can choose the different hyperparameters '
                             'or keep their default values. Default on False, write -training to make it true')
    parser.add_argument("-pred", "--only_prediction", action='store_false',
                        help=' | Make only one prediction on a randomly selected image'
                             ' in the seg_pred folder. Write -pred to activate it (you must also disable the plot '
                             'by writing -plot)')
    parser.add_argument("-y", "--models_path", type=str, required=False, default="./Models/Best_Model_trained",
                        metavar='')
    parser.add_argument("-kernel", "--kernel_size", type=int, default=3, required=False, choices=[1, 3, 5, 7],
                        help=" | Change the width and weight of the filters mask", metavar='')
    parser.add_argument("-cm", "--confusion_matrix", action='store_false',
                        help=" | Showing confusion matrix and classification report. Default True, write -cm to disable")
    parser.add_argument("-pat", "--patience", type=int, default=5, required=False, metavar='',
                        help=' | change the patience of the chekpoint during training')
    parser.add_argument('-height', "--height", type=int, default=150, required=False, help=" | change the height of "
                                                                                           "your images", metavar='')
    parser.add_argument('-width', "--width", type=int, default=150, required=False, help=" | change the width of "
                                                                                         "your images", metavar='')
    parser.add_argument('-keras', "--keras_data", action='store_false', help=' | train the model with'
                                                                                         ' our imported data or with'
                                                                                         ' keras data imported. By'
                                                                                         ' default on True which means'
                                                                                         ' using keras data')
    args = parser.parse_args()

    return args.data_train_path, args.data_test_path, args.data_pred_path, args.data_val_path, args.image_extension, \
           args.metrics, args.batch_size, args.epochs_number, args.filters, args.plot, args.evaluation,\
           args.training, args.only_prediction, args.models_path, args.kernel_size, args.confusion_matrix, \
           args.patience, args.height, args.width, args.keras_data


def plot_random_images(data_train: array, label: array, label_dictionary: dict) -> None:
    """
    :param data_train: numpy array including our training images
    :param label: a numpy array or a tensorflow tensor containing the label for each image of data_train, formed by an
     array of the length of the number of labels composed of 0 and 1, 1 being the label of the image
    :param label_dictionary: dictionary containing the labels and their numerical values
    """
    plt.figure(figsize=(10, 10))

    for i in range(25):
        image = np.random.randint(0, data_train.shape[0]-1)
        img_label = label_dictionary[label[image]]
        image = data_train[image]
        # the line below converts the last dimension of the image array from GBR to RGB
        image = image[..., ::-1]

        plt.subplot(5, 5, i + 1)
        plt.title(img_label)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(image)

    plt.show()

#todo : dans le rapport bien expliqué le réseau de neurones, le choix de son architecture !
# expliquer  TOUT ce qui est en rapport avec le réseau de neurones !
def functional_model(filters: int, kernel_size: int, data_train_path: str, height: int, width: int):
    """
    :param filters:
    :param kernel_size:
    :param data_train_path:
    :param height:
    :param width:
    :return:
    """
    # Shape and characteristics of our input data
    my_input = Input(shape=(height, width, 3))

    # First stack of layers
    x = Conv2D(filters, (kernel_size, kernel_size), activation="relu")(my_input)
    x = Conv2D(filters, (kernel_size, kernel_size), activation="relu")(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    # Second stack of layers
    x = Conv2D(filters * 2, (kernel_size, kernel_size), activation="relu")(x)
    x = Conv2D(filters * 2, (kernel_size, kernel_size), activation="relu")(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    # Third stack of layers
    x = Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu')(x)
    x = Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu')(x)
    x = Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    # Fourth stack of layers
    x = Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu')(x)
    x = Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu')(x)
    x = Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    # todo : make test with Flatten and GlobalAVGPool2D
    # x = Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(len(os.listdir(data_train_path)), activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

def add_checkpoint(models_path: str, patience: int):
    """
    :param models_path: file to which we want to save our data
    :param patience: Number of epochs without score increase before stopping the algorithm
    :return:
    """
    checkpoint_saver = ModelCheckpoint(
        models_path,
        monitor="val_accuracy",
        mode="max",
        save_freq="epoch",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(monitor="val_accuracy", patience=patience)

    return checkpoint_saver, early_stop


def random_prediction(model, data_pred_path: str, label_dictionary: dict, height: int, width: int) -> None:
    """
    :param model:
    :param data_pred_path:
    :param label_dictionary:
    :param height:
    :param width:
    :return:
    """

    image = random.choice(os.listdir(data_pred_path))

    image = tf.io.read_file(os.path.join(data_pred_path, image))
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (height, width))
    image = tf.expand_dims(image, axis=0)

    prediction = model.predict(image)
    prediction_index = np.argmax(prediction)

    fig = plt.imshow(image[0])
    fig.axes.set_axis_off()
    plt.title(f'{label_dictionary[prediction_index]}\nProbability of belonging to the label : '
              f'{round(prediction.max()*100, 2)}%')

    plt.show()

def confusion_matrix_and_classification_report(data_generator, model, data_path: str) -> None:
    """
    :param data_generator:
    :param model: last registered model or the model just trained
    :param data_path:
    :return:
    """

    print("\nMaking prediction on test set:")
    Y_pred = model.predict(data_generator, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)
    target_name = os.listdir(data_path)
    print(f'\nConfusion Matrix: \n{confusion_matrix(data_generator.classes, y_pred)}')
    print(f'\nClassification Report:\n {classification_report(data_generator.classes, y_pred, target_names=target_name)}')


if __name__ == '__main__':
    (data_train_path,
     data_test_path,
     data_pred_path,
     data_val_path,
     image_extension,
     metrics,
     batch_size,
     epochs_number,
     filters,
     plot,
     evaluation,
     training,
     only_prediction,
     models_path,
     kernel_size,
     confusion_matrix_activation,
     patience,
     height,
     width,
     keras_data) = parse_arguments()

# Creation of a dictionary of label containing the labels as well as the labels in numerical format
    label_dictionary = get_label_dictionary(data_train_path)

    if only_prediction:
    # Import of data with our own code, this allows us to display the images
        x_train, x_test, y_test, y_train, x_val, y_val = \
            data_preparation(data_train_path, data_test_path, data_val_path, image_extension, width, height)


    if plot:
    # Display 25 images randomly drawn from the dataset with their labels
        plot_random_images(x_train, y_train, label_dictionary)

    # Import of data via keras, this allows the dataset to be modified during training, improving the predictive
    # quality of the model
    train_generator, val_generator, test_generator = create_data_generator(batch_size=batch_size,
                                                                            data_train_path=data_train_path,
                                                                            data_val_path=data_val_path,
                                                                            data_test_path=data_test_path,
                                                                           height=height, width=width)




    if training:
    # Model training

        # Creation of checkpoints for training
        checkpoint_saver, early_stop = add_checkpoint(models_path, patience)

        # Importing the neural network model
        model = functional_model(filters, kernel_size, data_train_path, height, width)

        if keras_data:
            # Configures the model for training
            model.compile(optimizer="adam", loss='categorical_crossentropy', metrics="accuracy")
            # Training of the model on our data and the predefined and modifiable parameters (see parser_arguments function
            # above)
            model.fit(train_generator,
                      batch_size=batch_size,
                      epochs=epochs_number,
                      callbacks=[checkpoint_saver, early_stop],
                      validation_data=val_generator)
        else:
            model.compile(optimizer="adam", loss="SparseCategoricalCrossentropy", metrics='accuracy')

            data_val = (x_val, y_val)

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs_number,
                      callbacks=[checkpoint_saver, early_stop],
                      validation_data=data_val)


    if evaluation:
        # Evaluation
        model = tf.keras.models.load_model(models_path)
        # Reminder of the neural network architecture
        model.summary()

        # print(f'Evaluating validation set: ')
        # model.evaluate(val_generator, batch_size=batch_size)

        print('Evaluating test set: ')
        if keras_data:
            model_evaluation = model.evaluate(test_generator, batch_size=batch_size)
        else:
            model_evaluation = model.evaluate(x_test, y_test, batch_size=batch_size)
        print(f'Loss compute on test prediction: {round(model_evaluation[0], 3)}\nAccuracy compute on test prediction '
              f'{round(model_evaluation[1], 3)}')

    if confusion_matrix_activation:
    # Display the confusion matrix and classification report of the model
        model = tf.keras.models.load_model(models_path)

        # Allows not to display the model structure a second time if the model is evaluated before
        if evaluation == False:
            model.summary()
        else:
            pass
        confusion_matrix_and_classification_report(test_generator, model=model, data_path=data_test_path)



# Prediction on a random pred_image
    # First we import the last registered model
    model = tf.keras.models.load_model(models_path)
    # then we making prediction on a randomly chosen image
    random_prediction(model, data_pred_path, label_dictionary, height, width)

# Allows you to iterate making a prediction on an image without having to restart the program
    while True:
        again = input("Do you want to make a prediction on a new random image? Enter y/n: ")

        if again == "n":
            quit()
        elif again == 'y':
            random_prediction(model, data_pred_path, label_dictionary, height, width)
        else:
            print("Enter \"y\" or \"n\".")
