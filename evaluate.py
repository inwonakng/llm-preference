from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
import json
import pandas as pd
import numpy as np

metrics = {
    'Accuracy': accuracy_score,
    'Balanced Accuracy': balanced_accuracy_score,
    'F1 Micro': lambda y_true, y_pred : f1_score(y_true, y_pred, average='micro')
}

true_labels = []
predicted_labels = []
for f in Path('output').glob('*.json'):
    result = json.load(open(f))
    true_labels += [result['true_label']]
    predicted_labels += [result['predicted_label']]

true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

print('#' * 40)
print('Task 1. Preference Detection')
print('#' * 40)
print()
y_true = true_labels > 0
y_pred = predicted_labels > 0
print(
    pd.DataFrame(
        [{
            name: m(y_true, y_pred)
            for name, m in metrics.items()
        }],
        index = ['Score'],
    )
)   
print()
print(
    pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels = [0,1]
        ),
        index = ['true:yes', 'true:no'],
        columns = ['pred:yes', 'pred:no']

    )
)
print()
print('#' * 40)
print('Task 2. Preference Classification')
print('#' * 40)
print()
y_true = true_labels[(true_labels > 0)&(predicted_labels > 0)] - 1
y_pred = predicted_labels[(true_labels > 0)&(predicted_labels > 0)] - 1 
print(
    pd.DataFrame(
        [{
            name: m(y_true, y_pred)
            for name, m in metrics.items()
        }],
        index = ['Score'],
    )
)
print()
print(
    pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels = [0,1,2]
        ),
        index = ['true:A>B', 'true:A<B', 'true:A=B'],
        columns = ['pred:A>B', 'pred:A<B', 'pred:A=B'],

    )
)
print()
print('#' * 40)
print('Task 3. 4-way Classification')
print('#' * 40)
print()
y_true = true_labels
y_pred = predicted_labels 
print(
    pd.DataFrame(
        [{
            name: m(y_true, y_pred)
            for name, m in metrics.items()
        }],
        index = ['Score'],
    )
)
print()
print(
    pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels = [0,1,2,3]
        ),
        index = ['true:A?B','true:A>B', 'true:A<B', 'true:A=B'],
        columns = ['pred:A?B','pred:A>B', 'pred:A<B', 'pred:A=B'],

    )
)
print()