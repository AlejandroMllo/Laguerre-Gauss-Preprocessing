from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
from joblib import load

from Functions.image_handling import find_files, load_images, load_videos, show_video, rgb2grayscale
from Functions.CropImage import CropImage
from Functions.kernels import laguerre_gauss_filter
from Functions.fourier_transform_pipeline import ft_pipeline


class App:

    def __init__(self, model, model_specs, is_joblib=False):
        # Init. App
        # Load Model
        # Read Data
        # Inference
        # Display Results
        if not is_joblib:
            self.model = load_model(model)
        else:
            self.model = load(model)
        self.model_specs = model_specs
        self.is_joblib = is_joblib

    def run(self, data_path, video=False):

        data = find_files(data_path, sorted_by_idx=False)

        if video:
            media = load_videos(data, data_path)
        else:
            media = load_images(data, data_path)

        if video:
            print(len(media), len(media[0]))

        size = self.model_specs['size']
        omega = self.model_specs['omega']

        cropper = CropImage(size)

        for instance in media:

            instance = rgb2grayscale(instance)

            plt.imshow(instance, cmap='gray')
            plt.title('Input')
            plt.show()

            original_crops = cropper.crop(instance)
            lg_filter = laguerre_gauss_filter(size, omega)
            ft_lg_filter = np.fft.fft2(lg_filter)
            x_profile, y_profile = ft_pipeline(ft_lg_filter, original_crops)
            ft_crops = np.abs(np.concatenate((x_profile, y_profile), axis=1))

            if not self.is_joblib:
                predictions = self.model.predict_on_batch(ft_crops)
            else:
                predictions = self.model.predict(ft_crops)

            print(predictions)

            pos_preds = []

            for i in range(len(predictions)):

                if not self.is_joblib:
                    pred = np.argmax(predictions[i])
                else:
                    pred = predictions[i]

                if pred == 0:
                    continue

                # img = original_crops[i]

                # plt.imshow(img, cmap='gray')
                # plt.title('Prediction: ' + str(pred))
                # plt.show()
                # plt.pause()
                # plt.close()

                pos_preds.append(original_crops[i])

            print(len(pos_preds))
            n_row, n_col = 2, 3
            _, axs = plt.subplots(n_row, n_col, figsize=(12, 12))
            axs = axs.flatten()
            plt.suptitle('Positive Predictions kNN')
            for img, ax in zip(pos_preds, axs):
                ax.imshow(img, cmap='gray')
            plt.show()


if __name__ == '__main__':

    model_specs = dict()
    model_specs['size'] = 64
    model_specs['omega'] = 0.9

    # model = '/home/alejandro/Documents/Universidad/Semestre 6/PI1/Models/Models/MLP_Ae_v0.1/'
    model = '/home/alejandro/Documents/Universidad/Semestre 6/PI1/Models/SupervisedModels/MLP_Ae_kNN(1)_augmented_v0.1'
    image_path = '/home/alejandro/Desktop/images/'
    video_path = '/home/alejandro/Desktop/videos/'

    # Execute
    app = App(model, model_specs, is_joblib=True)
    app.run(image_path, video=False)

