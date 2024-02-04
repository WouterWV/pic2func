# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
# add .legacy if you use tensorflow>XX
from tensorflow.keras.optimizers.legacy import SGD

# define cnn model
def define_model():
    """Define a simple CNN model for digit recognition.

    This model was trained on the MNIST dataset, somewhere in 2022. The
    model (parameters) were saved and loaded here.
    The stochastic gradient descent optimizer is now part of the legacy
    optimizers in tensorflow.keras.legacy.

    Returns
    -------
    model : tensorflow.keras.Sequential
        The model for digit recognition.

    """
    model = Sequential()
    model.add(Conv2D(32,
                     (3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def predict_digit(digit, model):
    """Predict a digit from a 28x28 picture.

    Parameters
    ----------
    digit : np.array
        A 28x28 array with pixel values.
    model : tensorflow.keras.Sequential
        The model for digit recognition.

    Returns
    -------
    int
        The predicted digit.

    """
    digit = np.array([digit])
    digit = digit.reshape((digit.shape[0], 28, 28, 1))
    pred = model.predict(digit)
    return np.argmax(pred[0])  # 0-9

def predict_tickvalues(Iticks, Jticks, tickvals, model):
    """Predict the tickvalues from the tickpics.

    Parameters
    ----------
    Iticks : list
        A list of tick values.
    Jticks : list
        A list of tick values.
    tickvals : list
        A list of tickpics.
    model : tensorflow.keras.Sequential
        The model for digit recognition.

    Returns
    -------
    IXticks : list
        A list of tick values and their predicted values.
    JYticks : list
        A list of tick values and their predicted values.

    """
    IXticks = []
    JYticks = []
    Itickpics = tickvals[:len(Iticks)]
    Jtickpics = tickvals[len(Iticks):]
    for tick, tickpics in zip(Iticks,Itickpics):
        val = ""
        for tickpic in tickpics:
            val = val+str(predict_digit(tickpic, model))
        IXticks.append([tick,int(val)])
    for tick, tickpics in zip(Jticks,Jtickpics):
        val = ""
        for tickpic in tickpics:
            val = val+str(predict_digit(tickpic, model))
        JYticks.append([tick,int(val)])

    return IXticks, JYticks
