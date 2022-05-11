"""NOTES: Batch data is different each time in keras, which result_each in slight differences in results."""
import argparse
import datetime
import matplotlib.pyplot as plt
import keras
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from argparse import ArgumentParser

from utils.utils import *
from utils.models import *

def train(args):
    x_train, y_train, groups_train, x_test, y_test, groups_test = load_data()
    print('x_train:', x_train.shape)
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    model = model6(input_shape=x_train.shape[1:])

    model.summary()

    os.makedirs("result_each/{}".format(args.model_name), exist_ok=True)
    plot_model(model, "result_each/{}/{}.png".format(args.model_name, args.model_name))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lr_schedule) # Dynamic adjustment learning rate

    start = datetime.datetime.now()
    history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test),
                        callbacks=[lr_scheduler])
    end = datetime.datetime.now()
    print('Training Time:', end-start)

    model.save("result_each/{}/{}.h5".format(args.model_name, args.model_name)) # Save training model

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('result_each/{}/{}_accuracy.png'.format(args.model_name, args.model_name))
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('result_each/{}/{}_loss.png'.format(args.model_name, args.model_name))
    plt.show()

    print("train num:", len(y_train))
    print("test num:", len(y_test))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True)

    args = parser.parse_args()
    train(args)