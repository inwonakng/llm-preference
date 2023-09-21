import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import click

from lang_pref.config.paths import(
    OUTPUT_PATH,
    DATA_PATH,
    FIGURES_PATH,
)

@click.command()
@click.option('--dataset', default='college_confidential', help='Name of dataset to use')
@click.option('--template', default='inwon', help='Name of template to use for prompts.')
@click.option('--bin_range', default=20, type=int, help='Range to bin the word count by.')
@click.option('--use_example', is_flag=True, help='Use example in prompt')
def evaluate(
    dataset: str,
    template: str,
    bin_range: int, 
    use_example: bool, 
):
    postfix = 'without_example'
    if use_example:
        postfix = 'with_example'
    results_dir = OUTPUT_PATH / 'llm' / dataset / template / postfix

    results = []
    df = pd.read_csv(DATA_PATH / dataset / 'dataset.csv')
    for f in results_dir.glob('*.json'):
        result = json.load(open(f))
        idx = int(f.stem)
        true_lab = result['true_label']
        pred_lab = result['predicted_label']
        # has_pref =
        results += [{
            'wordcount': len(df.iloc[idx]['text'].split()),
            'true_label': true_lab,
            'predicted_label': pred_lab,
        }]

    results = pd.DataFrame(results)

    true_lab_3_class = results['true_label'].values.copy()
    true_lab_3_class[true_lab_3_class == 3] = 0
    pred_lab_3_class = results['predicted_label'].values.copy()
    pred_lab_3_class[pred_lab_3_class == 3] = 0

    results['3_class_correct'] = true_lab_3_class == pred_lab_3_class
    results['4_class_correct'] = results['true_label'] == results['predicted_label'] 
    results.sort_values(by=['wordcount'])

    results['wordcount_bin'] = pd.cut(
        results['wordcount'],
        [
            i*bin_range 
            for i in range(results['wordcount'].max() // bin_range + 2)
        ],
        labels = [
            (i+1) * bin_range
            for i in range(results['wordcount'].max() // bin_range + 1)
        ]
    )

    by_bin = []

    for max_count, performance in results.groupby('wordcount_bin'):
        by_bin += [{
            'Max Wordcount': max_count,
            '3 Class Correct': performance['3_class_correct'].sum(),
            'row_count': len(performance),
            # '3_class_incorrect': len(performance) - performance['3_class_correct'].sum(),
            '4 Class Correct': performance['4_class_correct'].sum(),
            # '4_class_incorrect': len(performance) - performance['4_class_correct'].sum()
        }]
        # break
    by_bin = pd.DataFrame(by_bin)

    fig, axes = plt.subplots(nrows=2, figsize = (10,6))

    sns.barplot(
        data = by_bin,
        x = 'Max Wordcount',
        y = 'row_count',
        ax = axes[0],
        color = 'red',
    )

    sns.barplot(
        data = by_bin,
        x = 'Max Wordcount',
        y = '3 Class Correct',
        ax = axes[0],
        color='blue'
    ).set_yscale('log')

    for i, tick in enumerate(axes[0].get_xticklabels()):
        if i % 5 == 0:
            tick.set_visible(True)
        else:
            tick.set_visible(False)

    sns.barplot(
        data = by_bin,
        x = 'Max Wordcount',
        y = 'row_count',
        ax = axes[1],
        color = 'red',
    )

    sns.barplot(
        data = by_bin,
        x = 'Max Wordcount',
        y = '4 Class Correct',
        ax = axes[1],
        color='blue'
    ).set_yscale('log')


    for i, tick in enumerate(axes[1].get_xticklabels()):
        if i % 5 == 0:
            tick.set_visible(True)
        else:
            tick.set_visible(False)

    fig.suptitle(' '.join([w.capitalize() for w in f'{dataset}_{postfix}'.split('_')]))
    fig.tight_layout()

    fig_save_dir = FIGURES_PATH / 'llm' / dataset / postfix
    fig_save_dir.mkdir(parents=True, exist_ok= True)
    fig.savefig(fig_save_dir / 'eval_per_token.pdf', bbox_inches='tight')
    # fig.yscale('log')
    # ax.



if __name__ == '__main__':
    evaluate()
