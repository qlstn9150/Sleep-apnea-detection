import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
from keras.models import load_model
from scipy.interpolate import splev, splrep
from sklearn.metrics import confusion_matrix

from utils.utils import *


'''def load_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))

    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:
        apnea_ecg = pickle.load(f)

    x_train = []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1))
    y_train = np.array(y_train, dtype="float32")

    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)):
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return (x_train, y_train, groups_train), (x_test, y_test, groups_test)'''

model_name = 'LeNet'

if __name__ == "__main__":
    x_train, y_train, groups_train, x_test, y_test, groups_test = load_data()

    model = load_model("models/{}.h5".format(model_name))
    model.summary()

    #print("training:")
    #y_true, y_pred = y_train, np.argmax(model.predict(x_train, batch_size=1024, verbose=1), axis=-1)

    #C = confusion_matrix(y_true, y_pred, labels=(1, 0))
    #TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    #acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    #print("acc: {}, sn: {}, sp: {}".format(acc, sn, sp))

    print("testing:")
    y_true, y_pred = y_test, np.argmax(model.predict(x_test, batch_size=1024, verbose=1), axis=-1)
    #y_true = y_test
    #y_pred = model.predict(x_test)
    #output = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}) #, "subject": groups_test
    #output.to_csv("result/{}.csv".format(model_name), index=False)

    C = confusion_matrix(y_true, y_pred, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    acc, sn, sp = round(acc, 2), round(sn, 2), round(sp, 2)
    print("acc: {}\nsn: {}\nsp: {}".format(acc, sn, sp))

    with open("result/{}_roc.txt".format(model_name), "w") as f:
        f.write("{} | accuracy: {} | sensitivity: {} | specificity: {}".format(model_name, acc, sn, sp))

    '''# make ROC curve
    # save prediction score
    y_pred = model.predict(x_test)
    output = pd.DataFrame({"y_true": y_test[:, 1], "y_score": y_pred[:, 1], "subject": groups_test})
    output.to_csv("record/output/{}.csv".format(model_name), index=False)

    output = []
    df = pd.read_csv('record/output/{}.csv'.format(model_name), header=0)
    df["y_pred"] = df["y_score"] > 0.5
    df.name = model_name
    output.append(df)
    output = pd.concat(output, axis=1)

    FP, TP, thresholds = roc_curve(df["y_true"], df["y_score"])

    #output['y_pred'] = output['y_pred'] > 0.5
    #FP, TP, thresholds = roc_curve(output["y_true"], output["y_pred"])

    plt.plot(FP, TP, color='blue', label=model_name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.savefig('result/{}_roc.png'.format(model_name))
    plt.show()'''


