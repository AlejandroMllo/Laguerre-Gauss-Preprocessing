from Functions.image_handling import find_files, load_images
from Functions.kernels import laguerre_gauss_filter
from Functions.fourier_transform_pipeline import ft_pipeline, ft_pipeline_no_shift, ft_pipeline_no_transform

import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical


class GeometricShapes:

    def __init__(self, path):

        self.path = path

        self.shapes_type = ['circle', 'square', 'triangle']
        self.shapes_label = {'circle': 0, 'square': 1, 'triangle': 2}

    def get(self, split='train', preprocess='line_profile', categorical_labels=True):

        data = self.load_data(split)
        x, y = self.join_data(data)

        if preprocess == 'line_profile':
            x = self.get_line_profiles(x, x.shape[1])
        elif preprocess == 'flattened':
            x = x.reshape((x.shape[0], -1))
        else:
            x = x.reshape(x.shape[0], 64, 64, 1)

        if categorical_labels:
            y = to_categorical(y, len(self.shapes_type))

        x, y = shuffle(x, y)

        return x, y

    def load_data(self, split='train'):
        data = dict()

        for shape in self.shapes_type:
            path = self.path + '/' + str(split) + '/' + str(shape) + '/'
            imgs = find_files(path)
            x = load_images(imgs, path)
            y = [self.shapes_label[shape]] * len(x)
            data[shape] = (x, y)

        return data

    def join_data(self, dataset):

        shapes, labels = [], []

        for shape in self.shapes_type:
            x, y = dataset[shape]
            shapes.extend(x)
            labels.extend(y)

        shapes = np.array(shapes)
        labels = np.array(labels)

        return shapes, labels

    @staticmethod
    def get_line_profiles(x, size, omega=0.9):

        lg_filter = laguerre_gauss_filter(size, omega)
        ft_lg_filter = np.fft.fft2(lg_filter)

        x_profile, y_profile = ft_pipeline(ft_lg_filter, x)

        # ft_pipeline_no_shift(ft_lg_filter, x)
        # ft_pipeline_no_transform(ft_lg_filter, x)

        return np.abs(np.concatenate((x_profile, y_profile), axis=1))


def geometric_shapes(preprocess, splits=['train', 'validation'], categorical_labels=True):

    model_params = dict()
    model_params['num_classes'] = 3

    path = '/home/alejandro/Documents/Universidad/Semestre 6/PI1/Datasets/Mineria/64x64/geometric_shapes'
    shapes = GeometricShapes(path)

    data = dict()
    for spl in splits:
        spl_data = shapes.get(split=spl, preprocess=preprocess, categorical_labels=categorical_labels)
        model_params[spl + '_imbalance'] = np.unique(spl_data[1], axis=0, return_counts=True)
        data[spl] = spl_data

    return model_params, data


def geometric_shapes_line_profile(version, custom_name='', splits=['train', 'validation'], categorical_labels=True):

    model_params, data = geometric_shapes('line_profile', splits, categorical_labels=categorical_labels)

    custom_name = '' if str(custom_name) == '' else str(custom_name) + '_'
    model_params['name'] = 'MLP_GS_' + str(custom_name) + 'v' + str(version)
    model_params['input_dim'] = 128

    return model_params, data


def geometric_shapes_flattened(version, custom_name='', splits=['train', 'validation'], categorical_labels=True):

    model_params, data = geometric_shapes('flattened', splits, categorical_labels=categorical_labels)

    custom_name = '' if str(custom_name) == '' else str(custom_name) + '_'
    model_params['name'] = 'MLP_GS_' + str(custom_name) + 'v' + str(version)
    model_params['input_dim'] = 4096

    return model_params, data


def geometric_shapes_images(version, custom_name='', splits=['train', 'validation'], categorical_labels=True):

    model_params, data = geometric_shapes('images', splits, categorical_labels=categorical_labels)

    model_params['input_shape'] = (64, 64, 1)
    custom_name = '' if str(custom_name) == '' else str(custom_name) + '_'
    model_params['name'] = 'CNN_GS_' + str(custom_name) + 'v' + str(version)

    return model_params, data


if __name__ == '__main__':

    _, data = geometric_shapes_line_profile('', splits=['train'])

    x, y = data['train']

    y = np.argmax(y, axis=1)

    circle = []
    square = []
    triangle = []
    for i in range(len(y)):

        if y[i] == 0:
            circle.append(x[i])
        elif y[i] == 1:
            square.append(x[i])
        else:
            triangle.append(x[i])

    circle = np.array(circle)
    square = np.array(square)
    triangle = np.array(triangle)

    circle_mean = np.mean(circle, axis=0)
    circle_mean_x = circle_mean[:64]
    circle_mean_y = circle_mean[64:]

    square_mean = np.mean(square, axis=0)
    square_mean_x = square_mean[:64]
    square_mean_y = square_mean[64:]

    triangle_mean = np.mean(triangle, axis=0)
    triangle_mean_x = triangle_mean[:64]
    triangle_mean_y = triangle_mean[64:]

    import matplotlib.pyplot as plt

    x_range = np.arange(-32, 32)

    # plt.subplot(231)
    # plt.plot(x_range, negative_mean_x)
    #
    # plt.subplot(232)
    # plt.plot(x_range, positive_mean_x)

    plt.subplot(121)
    plt.title('Mean Line Profile x-axis')
    plt.scatter(x_range, circle_mean_x, c='r', marker='o', label='Circle')
    plt.xlabel('Position along x-axis')
    plt.ylabel('Amplitude')
    plt.scatter(x_range, square_mean_x, c='g', marker='s', label='Square')
    plt.xlabel('Position along x-axis')
    plt.ylabel('Amplitude')
    plt.scatter(x_range, triangle_mean_x, c='b', marker='^', label='Triangle')
    plt.xlabel('Position along x-axis')
    plt.ylabel('Amplitude')

    # plt.subplot(234)
    # plt.plot(x_range, negative_mean_y)
    #
    # plt.subplot(235)
    # plt.plot(x_range, positive_mean_y)

    plt.subplot(122)
    plt.title('Mean Line Profile y-axis')
    plt.scatter(x_range, circle_mean_y, c='r', marker='o', label='Circle')
    plt.xlabel('Position along y-axis')
    plt.ylabel('Amplitude')
    plt.scatter(x_range, square_mean_y, c='g', marker='s', label='Square')
    plt.xlabel('Position along y-axis')
    plt.ylabel('Amplitude')
    plt.scatter(x_range, triangle_mean_y, c='b', marker='^', label='Triangle')
    plt.xlabel('Position along y-axis')
    plt.ylabel('Amplitude')

    plt.legend()
    plt.show()


