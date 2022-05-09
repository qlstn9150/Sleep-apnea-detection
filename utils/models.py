from keras import layers
from keras.models import Input, Model
from keras.regularizers import l2

def LeNet(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.Dropout(0.8)(x) # Avoid overfitting

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def model1(input_shape, weight=1e-3):
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(16, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
                      kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
                      kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.Dropout(0.8)(x) # Avoid overfitting

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def model2(input_shape, weight=1e-3):
    inputs = Input(shape=input_shape)

    x = layers.GRU(16)
    x = layers.Dropout(0.8)(x) # Avoid overfitting

    x = layers.GRU(32)
    x = layers.Dropout(0.8)(x)  # Avoid overfitting

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def model3(input_shape, weight=1e-3):
    inputs = Input(shape=input_shape)

    x = layers.ConvLSTM1D(32)(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM1D(32)(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM1D(32)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model4(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x1 = layers.Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x1 = layers.MaxPooling1D(pool_size=3)(x1)

    x2 = layers.Conv1D(64, kernel_size=5, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x1)
    x2 = layers.MaxPooling1D(pool_size=3)(x2)

    #x2 = layers.Concatenate([x1, x2])

    x = layers.Dropout(0.8)(x2)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def model5(input_shape):
    inputs = Input(shape=input_shape)

    x1 = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal")(inputs)
    x1 = layers.MaxPooling1D(pool_size=3)(x1)

    x2 = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal")(
        x1)
    x2 = layers.MaxPooling1D(pool_size=3)(x2)

    x2 = layers.Add([x1, x2])

    x3 = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu",
                       kernel_initializer="he_normal")(
        x2)
    x3 = layers.MaxPooling1D(pool_size=3)(x3)

    x3 = layers.Add([x1, x2, x3])

    x = layers.Flatten()(x3)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model6(input_shape):
    inputs = Input(shape=input_shape)

    x = layers.LSTM(64)(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(32)(x)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(16)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model7(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=3, strides=2, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.GRU(32)(x)

    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model8(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same", kernel_initializer="he_normal")(inputs)
    x = layers.PReLU()(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.Dropout(0.8)(x)

    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(32)(x)
    x = layers.PReLU()(x)

    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model