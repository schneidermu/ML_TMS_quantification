import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


def plot_confusion_matrix(clf, X_train, X_test, y_train, y_test):
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    classes = sorted(y_test.unique())
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred),
                               index=[f'actual {i}' for i in classes],
                               columns=[f'predicted {i}' for i in classes])
    display(conf_matrix)
    for i in classes:
        print(f"{i} precision:", conf_matrix.iloc[i, i]
              / np.sum(conf_matrix.iloc[i, :])*100)
        print(f"{i} recall:", conf_matrix.iloc[i, i]
              / np.sum(conf_matrix.iloc[:, i])*100)
