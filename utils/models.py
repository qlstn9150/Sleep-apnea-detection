from keras import layers, Sequential
from keras.models import Input, Model
from keras.regularizers import l2
from utils.utils import im2col

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


def LeNet2(input_shape, weight=1e-3):
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = layers.PReLU()(x)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.Dropout(0.8)(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = layers.PReLU()(x)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def LeNet3(input_shape, weight=1e-3):
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation='relu', kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation='relu', kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation='relu',  kernel_initializer="he_normal",
                      kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    #x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def LeNet4(input_shape, weight=1e-3):
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation='relu', kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation='relu', kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv1D(16, kernel_size=3, strides=2, padding="valid", activation='relu',  kernel_initializer="he_normal",
                      kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    #x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

'''def LeNet5(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(inputs)

    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model'''

def LeNet5(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(32))(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = layers.MaxPooling1D(pool_size=3)(x)


    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def LeNet6(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(64))(x)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(32))(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model




'''def LeNet9(input_shape, weight=1e-3):
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=3, strides=1, padding="valid", kernel_initializer="he_normal",
                      kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.TimeDistributed(layers.Dense(64))(x)

    x = layers.Conv1D(32, kernel_size=3, strides=1, padding="valid", kernel_initializer="he_normal",
                      kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.TimeDistributed(layers.Dense(32))(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model'''



def LeNet9(input_shape, weight=1e-3):
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", kernel_initializer="he_normal",
                      kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", kernel_initializer="he_normal",
                      kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = layers.TimeDistributed(layers.Dense(32))(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def model1(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(64))(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def model2(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(32))(x)

    '''x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = layers.MaxPooling1D(pool_size=3)(x)'''

    x = layers.TimeDistributed(layers.Dense(64))(x)

    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def model3(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(64))(x)

    '''x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = layers.MaxPooling1D(pool_size=3)(x)'''

    #x = layers.TimeDistributed(layers.Dense(64))(x)

    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model4(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    '''x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)'''


    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(64))(x)

    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model5(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    '''x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)'''


    x = layers.Conv1D(128, kernel_size=5, strides=2, padding="valid", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)

    #x = layers.TimeDistributed(layers.Dense(64))(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", kernel_initializer="he_normal",
                      kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.8)(x)
    x = layers.MaxPooling1D(pool_size=3)(x)

    #x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model6(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(32))(x)
    x = layers.Dropout(0.8)(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(64))(x)
    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def model7(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(64))(x)
    x = layers.Dropout(0.8)(x)

    x = layers.Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = layers.MaxPooling1D(pool_size=3)(x)

    x = layers.TimeDistributed(layers.Dense(32))(x)
    x = layers.Dropout(0.8)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
