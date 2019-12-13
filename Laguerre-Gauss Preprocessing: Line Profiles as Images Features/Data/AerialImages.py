from Functions.image_handling import find_files, load_images
from Functions.kernels import laguerre_gauss_filter
from Functions.fourier_transform_pipeline import ft_pipeline, ft_pipeline_no_shift, ft_pipeline_no_transform

import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import normalize


class AerialImages:

    def __init__(self, path):

        self.path = path

        images_path = find_files(self.path)
        self.images_path = shuffle(images_path)

    def get(self, split='train', preprocess='line_profile', categorical_labels=True):

        x, y = self.load_data(split)

        if preprocess == 'line_profile':
            x = self.get_line_profiles(x, x.shape[1])
        elif preprocess == 'flattened':
            x = x.reshape((x.shape[0], -1))
        elif preprocess == 'images':
            n, h, w = x.shape
            x = x.reshape((n, h, w, 1))
            x = x.astype('float32')
            x /= 255.0

        if categorical_labels:
            num_classes = len(np.unique(y))
            y = to_categorical(y, num_classes)

        x = normalize(x)

        return x, y

    def load_data(self, split='train'):

        # images_path = find_files(self.path)
        #
        # images_path = shuffle(images_path)

        num_samples = len(self.images_path)
        if split == 'train':
            images_path = self.images_path[: int(0.7 * num_samples)]
        elif split == 'validation':
            images_path = self.images_path[int(0.7 * num_samples): int(0.85 * num_samples)]
        elif split == 'test':
            images_path = self.images_path[int(0.85 * num_samples):]
        else:
            images_path = self.images_path

        labels = []

        for img in images_path:
            idx = 0
            for c in img:
                if c.isdigit():
                    break
                else:
                    idx += 1
            # labels.append(int(img[17]))
            labels.append(int(img[idx]))

        images = np.array(load_images(images_path, self.path))
        labels = np.array(labels)

        return images, labels

    @staticmethod
    def get_line_profiles(x, size, omega=0.9):

        lg_filter = laguerre_gauss_filter(size, omega)
        ft_lg_filter = np.fft.fft2(lg_filter)

        x_profile, y_profile = ft_pipeline(ft_lg_filter, x)

        # ft_pipeline_no_shift(ft_lg_filter, x)
        # ft_pipeline_no_transform(ft_lg_filter, x)

        return np.abs(np.concatenate((x_profile, y_profile), axis=1))


def aerial_images(preprocess, splits=['train', 'validation'], path=None):

    model_params = dict()
    model_params['num_classes'] = 2

    if path is None:
        path = '../sample_test_data/'
    shapes = AerialImages(path)

    data = dict()
    for spl in splits:
        spl_data = shapes.get(split=spl, preprocess=preprocess)
        model_params[spl + '_imbalance'] = np.unique(spl_data[1], axis=0, return_counts=True)
        data[spl] = spl_data

    return model_params, data


def aerial_images_line_profile(version, custom_name='', splits=['train', 'validation']):

    model_params, data = aerial_images('line_profile', splits)

    custom_name = '' if str(custom_name) == '' else str(custom_name) + '_'
    model_params['name'] = 'MLP_Ae_' + str(custom_name) + 'v' + str(version)
    model_params['input_dim'] = 128

    return model_params, data


def aerial_images_flattened(version, custom_name='', splits=['train', 'validation']):

    model_params, data = aerial_images('flattened', splits)

    custom_name = '' if str(custom_name) == '' else str(custom_name) + '_'
    model_params['name'] = 'MLP_Ae_' + str(custom_name) + 'v' + str(version)
    model_params['input_dim'] = 4096

    return model_params, data


def aerial_images_images(version, data_path, custom_name='', splits=['train', 'validation']):

    model_params, data = aerial_images('images', splits, data_path)

    custom_name = '' if str(custom_name) == '' else str(custom_name) + '_'
    model_params['name'] = 'CNN_Ae_' + str(custom_name) + 'v' + str(version)
    model_params['input_shape'] = (64, 64, 1)

    return model_params, data


if __name__ == '__main__':

    _, data = aerial_images_line_profile('', splits=['train'])

    x, y = data['train']

    y = np.argmax(y, axis=1)

    negative = []
    positive = []
    for i in range(len(y)):

        if y[i] == 0:
            negative.append(x[i])
        else:
            positive.append(x[i])

    negative = np.array(negative)
    positive = np.array(positive)

    negative_mean = np.mean(negative, axis=0)
    negative_mean_x = negative_mean[:64]
    negative_mean_y = negative_mean[64:]

    positive_mean = np.mean(positive, axis=0)
    positive_mean_x = positive_mean[:64]
    positive_mean_y = positive_mean[64:]

    import matplotlib.pyplot as plt

    x_range = np.arange(-32, 32)

    # plt.subplot(231)
    # plt.plot(x_range, negative_mean_x)
    #
    # plt.subplot(232)
    # plt.plot(x_range, positive_mean_x)

    plt.subplot(121)
    plt.title('Mean Line Profile x-axis')
    plt.scatter(x_range, negative_mean_x, c='r', marker='.', s=10, label='Class 0')
    plt.xlabel('Position along x-axis')
    plt.ylabel('Amplitude')
    plt.scatter(x_range, positive_mean_x, c='g', marker='+', s=10, label='Class 1')
    plt.xlabel('Position along x-axis')
    plt.ylabel('Amplitude')

    # plt.subplot(234)
    # plt.plot(x_range, negative_mean_y)
    #
    # plt.subplot(235)
    # plt.plot(x_range, positive_mean_y)

    plt.subplot(122)
    plt.title('Mean Line Profile y-axis')
    plt.scatter(x_range, negative_mean_y, c='r', marker='.', s=10, label='Class 0')
    plt.xlabel('Position along y-axis')
    plt.ylabel('Amplitude')
    plt.scatter(x_range, positive_mean_y, c='g', marker='+', s=10, label='Class 1')
    plt.xlabel('Position along y-axis')
    plt.ylabel('Amplitude')

    plt.legend()
    plt.show()
