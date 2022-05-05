from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Input, Model
from keras.regularizers import l2

def LeNet(input_shape, weight=1e-3):
	#Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = MaxPooling1D(pool_size=3)(x)

    x = Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = MaxPooling1D(pool_size=3)(x)

    x = Dropout(0.8)(x) # Avoid overfitting

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def model1(input_shape, weight=1e-3):
    # Create a Modified LeNet-5 model
    inputs = Input(shape=input_shape)

    x = Conv1D(32, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight))(inputs)

    x = MaxPooling1D(pool_size=3)(x)

    x = Conv1D(64, kernel_size=5, strides=2, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(1e-3), bias_regularizer=l2(weight))(x)

    x = MaxPooling1D(pool_size=3)(x)

    x = Dropout(0.8)(x)  # Avoid overfitting

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(2, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
