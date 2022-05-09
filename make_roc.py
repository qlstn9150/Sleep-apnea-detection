import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

model_list = ['LeNet', 'model1', 'model2', 'model3', 'model4']
color_list = list(mcolors.BASE_COLORS)
i=0
for model_name in model_list:
    output = []
    df = pd.read_csv('result_each/{}/{}.csv'.format(model_name, model_name), header=0)
    df["y_pred"] = df["y_score"] > 0.5
    df.name = model_name
    output.append(df)
    output = pd.concat(output, axis=1)

    FP, TP, thresholds = roc_curve(df["y_true"], df["y_score"])

    plt.plot(FP, TP, label=model_name, color=color_list[i])
    plt.legend()
    i += 1

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.savefig('result_all/roc.png'.format(model_name))
plt.show()
