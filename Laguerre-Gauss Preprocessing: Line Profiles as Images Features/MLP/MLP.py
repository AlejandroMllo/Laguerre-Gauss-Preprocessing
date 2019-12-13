from __future__ import print_function

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from datetime import datetime

from Data.GeometricShapes import geometric_shapes_line_profile, geometric_shapes_flattened
from Data.AerialImages import aerial_images_line_profile, aerial_images_flattened


def build_model():

    model = Sequential(name=model_name)
    model.add(Dense(128, activation='relu', input_dim=input_dim, name='input_layer'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', name='hidden_layer_1'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation='relu', name='hidden_layer_2'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax', name='output_layer'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(),
        metrics=['accuracy']
    )

    return model


def train_model(model, data, hyperparameters, save=True):

    train_data, val_data = data['train'], data['validation']

    x_train, y_train = train_data
    x_val, y_val = val_data

    epochs = hyperparameters['epochs']
    batch_size = hyperparameters['batch_size']

    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_val, y_val)
    )

    print(model.summary())
    if save:
        save_model(model, model_name)

    return model


def change_output_layer(model, new_num_classes):

    layers = model.layers[:-1]
    model = Sequential(name=model_name)
    for l in layers:
        model.add(l)
    model.add(Dense(new_num_classes, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(),
        metrics=['accuracy']
    )

    return model


def evaluate(model, data):

    x_val, y_val = data

    preds = model.predict(x_val)
    preds = np.argmax(preds, axis=1)
    y_val = np.argmax(y_val, axis=1)

    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average=None)

    print('ACC', acc)
    print('F1', f1)


if __name__ == '__main__':

    model_params, data = aerial_images_line_profile('0.1', splits=['all'], custom_name='')
    # model_params, data = aerial_images_line_profile('0.1', splits=['all'], custom_name='NoTransform')
    # model_params, data = aerial_images_line_profile('0.1', splits=['all'], custom_name='NoShift')
    # model_params, data = aerial_images_flattened('0.1', splits=['all'], custom_name='Flattened')

    model_name = model_params['name']
    input_dim = model_params['input_dim']
    num_classes = model_params['num_classes']

    print(model_params)

    # model = build_model()
    model = load_model(model_name)
    # model = change_output_layer(model, num_classes)

    # model_hyperparameters = dict()
    # model_hyperparameters['epochs'] = 256
    # model_hyperparameters['batch_size'] = 64

    # model = train_model(model, data, model_hyperparameters, save=True)

    evaluate(model, data['all'])