import pandas as pd
import keras
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from argparse import ArgumentParser

from utils.utils import *


def test(args):
    x_train, y_train, groups_train, x_test, y_test, groups_test = load_data()

    model = load_model("result_each/{}/{}.h5".format(args.model_name, args.model_name))
    model.summary()

    print("testing:")
    y_true, y_pred = y_test, np.argmax(model.predict(x_test, batch_size=1024, verbose=1), axis=-1)

    C = confusion_matrix(y_true, y_pred, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    acc, sn, sp = round(acc*100, 2), round(sn*100, 2), round(sp*100, 2)
    print("acc: {}\nsn: {}\nsp: {}".format(acc, sn, sp))

    with open("result_all/performance.txt".format(args.model_name), "a") as f:
        f.write("{} | {} | {} | {}".format(args.model_name, acc, sn, sp))
        f.write('\n')

    # save prediction score
    y_score = model.predict(x_test)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)
    output = pd.DataFrame({"y_true": y_test[:, 1], "y_score": y_score[:, 1], "subject": groups_test})
    output.to_csv('result_each/{}/{}.csv'.format(args.model_name, args.model_name), index=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', required=True)

    args = parser.parse_args()
    test(args)


