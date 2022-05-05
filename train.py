"""NOTES: Batch data is different each time in keras, which result in slight differences in results."""
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.utils import plot_model

from utils.utils import *
from utils.models import *

model_name = 'LeNet'

if __name__ == "__main__":
    x_train, y_train, groups_train, x_test, y_test, groups_test = load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes=2) # Convert to two categories
    #y_test = keras.utils.to_categorical(y_test, num_classes=2)

    model = LeNet(input_shape=x_train.shape[1:])
    model.summary()

    plot_model(model, "models/{}.png".format(model_name))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lr_schedule) # Dynamic adjustment learning rate
    #tensorboard = TensorBoard(log_dir='tensorboard/{}'.format(model_name))

    history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test),
                        callbacks=[lr_scheduler])

    model.save(os.path.join("models", "{}.h5".format(model_name))) # Save training model

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('result/{}_accuracy.png'.format(model_name))
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('result/{}_loss.png'.format(model_name))
    plt.show()

    print("train num:", len(y_train))
    print("test num:", len(y_test))