import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt

model_name = 'LeNet'

output = []
df = pd.read_csv('record/output/{}.csv'.format(model_name), header=0)
df["y_pred"] = df["y_score"] > 0.5
df.name = model_name
output.append(df)
output = pd.concat(output, axis=1)

FP, TP, thresholds = roc_curve(df["y_true"], df["y_score"])

plt.plot(FP, TP, color='blue', label=model_name)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.savefig('result/{}_roc.png'.format(model_name))
plt.show()
