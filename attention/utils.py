import torch
from torch import nn
from torch.autograd import Variable
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_score, f1_score
import os
from datasets import load_dataset, DatasetDict, Dataset

import numpy as np
from tqdm import tqdm

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def metrics(y_pred, y_true, labels):
    # micro = precision_score(y_true,y_pred,labels=labels,average='micro')
    weighted = f1_score(y_true, y_pred, average="weighted")
    # sample = precision_score(y_true,y_pred,average='samples')
    micro = f1_score(y_true, y_pred, labels=labels, average="micro")
    f1s = f1_score(y_true, y_pred, labels=labels, average=None)

    result = {"weighted": weighted, "mirco": micro}
    for i, sco in enumerate(f1s):
        result[f"F1[{i}]"] = sco

    return result


def get_metrics(y_true, y_pred, n_classes):
    indiv_f1s = f1_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
    )
    return [
        {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "F1 Micro": f1_score(y_true, y_pred, average="micro"),
            "F1 Macro": f1_score(y_true, y_pred, average="macro"),
            "F1 Weighted": f1_score(y_true, y_pred, average="weighted"),
            **{f"F1[{i}]": sco for i, sco in enumerate(indiv_f1s)},
        }
    ]


def show_3_way(true_labels, predicted_labels):
    print("Task 4. 3-way Classification")
    print("#" * 40)
    print()
    y_true = true_labels.copy()
    y_true[y_true == 3] = 0
    y_pred = predicted_labels.copy()
    y_pred[y_pred == 3] = 0
    metrics = pd.DataFrame(
        get_metrics(y_true, y_pred, 3),
        index=["Score"],
    )
    print(metrics)

    return metrics
