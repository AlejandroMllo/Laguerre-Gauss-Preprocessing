from sklearn import neighbors
import numpy as np
from joblib import dump, load
from sklearn.metrics import accuracy_score, f1_score

from Data.AerialImages import aerial_images_line_profile, aerial_images_flattened
from Data.GeometricShapes import geometric_shapes_line_profile, geometric_shapes_flattened
from Data.MNIST import load_mnist


def train(max_num_neighbors=1, save=False):

    for i in range(1, max_num_neighbors + 1):
        clf = neighbors.KNeighborsClassifier(i)
        clf.fit(X_train, y_train)

        train_score = clf.score(X_train, y_train)
        val_score = clf.score(X_val, y_val)

        print('------', i, '--------')
        print('TRAIN SCORE', train_score)
        print('VAL SCORE', val_score)

        if save:
            dump(clf, model_params['name'])


def model_test():

    clf = load(model_params['name'])

    for i in range(len(splits)):
        split_name = str(splits[i]).capitalize()

        print('** Results on:', split_name, 'data set.')

        input_data = X[i]
        output_data = y[i]

        pred = clf.predict(input_data)

        accuracy = accuracy_score(output_data, pred)
        f1 = f1_score(output_data, pred, average=None)

        print('\tAccuracy:', accuracy)
        print('\tF1:', f1)


if __name__ == '__main__':

    splits = ['train', 'validation', 'test', 'all']
    model_params, data = aerial_images_line_profile('0.1', splits=splits, custom_name='kNN(1)')
    print(model_params)

    X_train, y_train = data['train']
    X_val, y_val = data['validation']
    X_test, y_test = data['test']
    X_all, y_all = data['all']

    X = [X_train, X_val, X_test, X_all]
    y = []
    temp_y = [y_train, y_val, y_test, y_all]

    for labels in temp_y:
        y.append(np.argmax(labels, axis=1))

    # train(max_num_neighbors=1, save=False)
    model_test()