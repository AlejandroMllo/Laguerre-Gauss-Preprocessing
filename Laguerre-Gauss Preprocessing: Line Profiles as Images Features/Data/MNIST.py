from tensorflow.keras.datasets import mnist
import tensorflow.keras as keras
import numpy as np

from Functions.kernels import laguerre_gauss_filter
from Functions.fourier_transform_pipeline import ft_pipeline


def load_mnist(version, preprocess, categorical_labels=True, custom_name=''):

    num_classes = 10
    image_size = 28

    model_params = dict()
    model_params['num_classes'] = num_classes

    custom_name = '' if str(custom_name) == '' else str(custom_name) + '_'
    model_params['name'] = 'MLP_MNIST_' + str(custom_name) + 'v' + str(version)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if preprocess == 'line_profile':
        model_params['input_dim'] = image_size * 2

        lg_28 = laguerre_gauss_filter(image_size, 0.9)
        ft_lg_28 = np.fft.fft2(lg_28)

        x_pr_train, y_pr_train = ft_pipeline(ft_lg_28, x_train)
        x_pr_test, y_pr_test = ft_pipeline(ft_lg_28, x_test)

        x_train = np.abs(np.concatenate((x_pr_train, y_pr_train), axis=1))
        x_test = np.abs(np.concatenate((x_pr_test, y_pr_test), axis=1))
    elif preprocess == 'flattened':
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        model_params['input_dim'] = image_size ** 2
    else:
        img_rows, img_cols = image_size, image_size
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        model_params['input_shape'] = input_shape
        model_params['name'] = 'CNN_MNIST_' + str(custom_name) + 'v' + str(version)

    if categorical_labels:
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    splits = [('train', x_train, y_train), ('validation', x_test, y_test)]
    data = dict()
    for spl in splits:
        split, features, labels = spl
        model_params[split + '_imbalance'] = np.unique(labels, axis=0, return_counts=True)
        data[split] = (features, labels)

    return model_params, data
