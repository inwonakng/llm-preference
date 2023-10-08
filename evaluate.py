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
import click


def get_metrics(y_true, y_pred, n_classes):
    indiv_f1s = f1_score(y_true,y_pred,labels=list(range(n_classes)),average=None,zero_division=0)
    return [{
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'F1 Micro': f1_score(y_true, y_pred, average='micro'),
        'F1 Macro': f1_score(y_true, y_pred, average='macro'),
        'F1 Weighted': f1_score(y_true, y_pred, average='weighted'),
        **{
            f'F1[{i}]': sco
            for i,sco in enumerate(indiv_f1s)
        }
    }]

@click.command()
@click.option('--dataset', default='college_confidential', help='Name of dataset to use')
@click.option('--model', default='llama2-13b', help='Name of model to use')
@click.option('--template', default='inwon', help='Name of template to use for prompts.')
@click.option('--use_example', is_flag=True, help='Use example in prompt')
@click.option('--mode', default='textgen', help='Mode to use')
def evaluate(
    dataset: str,
    model: str,
    template: str,
    use_example: bool, 
    mode: str,
):
    if mode == 'openai':
        model = 'openai'

    results_dir = Path('output') / model / dataset / template
    eval_dir = Path('evaluation') / model / dataset / template
    if use_example:
        results_dir /= 'with_example_' + mode
        eval_dir /= 'with_example_' + mode
    else:
        results_dir /= 'without_example_' + mode
        eval_dir /= 'without_example_' + mode
    eval_dir.mkdir(exist_ok=True, parents=True)

    true_labels = []
    predicted_labels = []
    for f in results_dir.glob('*.json'):
        result = json.load(open(f))
        true_labels += [result['true_label']]
        predicted_labels += [result['predicted_label']]

    file_count = len(true_labels)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    print('#' * 40)
    print('Task 1. Preference Detection')
    print('#' * 40)
    print()
    y_true = true_labels > 0
    y_pred = predicted_labels > 0
    metrics = pd.DataFrame(
        get_metrics(y_true, y_pred, 2),
        index = ['Score'],
    )
    print(metrics)  
    metrics.to_csv(eval_dir / 'stage_1.csv')
    print()
    conf_mat = pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels = [0,1]
        ),
        index = ['true:yes', 'true:no'],
        columns = ['pred:yes', 'pred:no']

    )
    print(conf_mat)
    conf_mat.to_csv(eval_dir / 'stage_1_cm.csv')
    print()
    print('#' * 40)
    print('Task 2. Preference Classification')
    print('#' * 40)
    print()
    y_true = true_labels[(true_labels > 0)&(predicted_labels > 0)] - 1
    y_pred = predicted_labels[(true_labels > 0)&(predicted_labels > 0)] - 1 
    metrics = pd.DataFrame(
        get_metrics(y_true, y_pred, 3),
        index = ['Score'],
    )
    print(metrics)
    metrics.to_csv(eval_dir / 'stage_2.csv')
    print()
    conf_mat = pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels = [0,1,2]
        ),
        index = ['true:A>B', 'true:A<B', 'true:A=B'],
        columns = ['pred:A>B', 'pred:A<B', 'pred:A=B'],

    )
    print(conf_mat) 
    conf_mat.to_csv(eval_dir / 'stage_2_cm.csv')
    print()
    print('#' * 40)
    print('Task 3. 4-way Classification')
    print('#' * 40)
    print()
    y_true = true_labels
    y_pred = predicted_labels 
    metrics = pd.DataFrame(
        get_metrics(y_true, y_pred, 4),
        index = ['Score'],
    )
    print(metrics)
    metrics.to_csv(eval_dir / '4_way.csv')
    print()
    conf_mat = pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels = [0,1,2,3]
        ),
        index = ['true:A?B','true:A>B', 'true:A<B', 'true:A=B'],
        columns = ['pred:A?B','pred:A>B', 'pred:A<B', 'pred:A=B'],

    )
    print(conf_mat)
    conf_mat.to_csv(eval_dir / '4_way_cm.csv')
    print()
    print('#' * 40)
    print('Task 4. 3-way Classification')
    print('#' * 40)
    print()
    y_true = true_labels.copy()
    y_true[y_true == 3] = 0
    y_pred = predicted_labels.copy()
    y_pred[y_pred == 3] = 0
    metrics = pd.DataFrame(
        get_metrics(y_true, y_pred, 3),
        index = ['Score'],
    )
    print(metrics)
    metrics.to_csv(eval_dir / '3_way.csv')
    print()
    conf_mat = pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels = [0,1,2]
        ),
        index = ['true:A?B','true:A>B', 'true:A<B'],
        columns = ['pred:A?B','pred:A>B', 'pred:A<B'],

    )
    print(conf_mat)
    conf_mat.to_csv(eval_dir / '3_way_cm.csv')
    print()

    print(f'Number of data points: {file_count}')

if __name__ == '__main__':
    evaluate()