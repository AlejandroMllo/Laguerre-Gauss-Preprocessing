import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt

from Functions.image_handling import find_files, load_images, rgb2grayscale


class CropImage:

    def __init__(self, side=32):

        self.side = side

    def crop(self, image, resize=True):
        """
        :param image: Grayscale image.
        :param resize:
        :param save:
        :return:
        """

        image = np.array(image)
        h, w = image.shape

        images = []

        # images += [image[y: y + h // 2, x: x + w // 2] for x in range(0, w - (w % 2), w // 2) for y in range(0, h - (h % 2), h // 2)]  # 4
        # images += [image[y: y + h // 2, x: x + w // 3] for x in range(0, w - (w % 3), w // 3) for y in range(0, h - (h % 2), h // 2)]  # 6
        # images += [image[y: y + h // 3, x: x + w // 2] for x in range(0, w - (w % 2), w // 2) for y in range(0, h - (h % 3), h // 3)]  # 6
        # images += [image[y: y + h // 2, x: x + 3 * (w // 4)] for x in range(0, w - (3 - 1) * (w // 4), w // 4) for y in range(0, h - (h % 2), h // 2)]  # 4
        # images += [image[y: y + 3 * (h // 4), x: x + w // 2] for x in range(0, w - (w % 2), w // 2) for y in range(0, h - (3 - 1) * (h // 4), h // 4)]  # 4
        m, n = 256, 256
        images += [image[y: y + n, x: x + m] for x in range(0, w, m) for y in range(0, h, n)]
        # if h < w:
        #    images += [image[:, x: x + h] for x in range(0, w - h, w // 10)]
        # else:
        #    images += [image[y: y + w, :] for y in range(0, h - w, h // 10)]

        if resize:
            images = self.resize(images)

        return images

    def resize(self, images):

        resized = []

        dim = (self.side, self.side)
        for img in images:
            resized.append(resize(img, dim))

        return resized


if __name__ == '__main__':

    cropper = CropImage(64)

    path = '/home/alejandro/Documents/Universidad/Semestre 6/PI1/Datasets/Mineria/samples/'
    save_path = '/home/alejandro/Documents/Universidad/Semestre 6/PI1/Datasets/Mineria/crop_64x64_samples/'

    cropper = CropImage(64)

    imgs = find_files(path)
    print('found_images')
    images = load_images(imgs, path)
    print('loaded_images')

    count_0 = 0
    count_1 = 0
    for i in range(len(imgs)):
        image = rgb2grayscale(images[i])
        crops = cropper.crop(image)
        if i % 100 == 0:
            print(i)
        for c in crops:
            img_class = imgs[i][0]
            if img_class == 0:
                class_count = count_0
                count_0 += 1
            else:
                class_count = count_1
                count_1 += 1
            plt.imsave(save_path + str(img_class) + '_' + str(class_count) + '.png', c, cmap='gray')
