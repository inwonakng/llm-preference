import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import click
from pathlib import Path
BASE_PATH = Path(__file__).parent.parent.parent

def get_by_bin(results_dir, df, bin_range):
        
    results = []    
    for f in results_dir.glob('*.json'):
        result = json.load(open(f))
        if len(result.keys()) == 3: 
            idx = int(f.stem) #outputs only the name of the final component of your path without the suffix
            true_lab = result['true_label']
            pred_lab = result['predicted_label']
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
            # 'Max Wordcount': max_count,
            'Max Wordcount': max_count,
            'row_count': len(performance),
            '3 Class Correct': performance['3_class_correct'].sum(),
            '4 Class Correct': performance['4_class_correct'].sum(),
        }]

    by_bin = pd.DataFrame(by_bin)

    return by_bin

@click.command()
@click.option('--dataset', default='college_confidential_clean', help='Name of dataset to use')
@click.option('--template', default='3satge', help='Name of template to use for prompts.')
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

    df = pd.read_csv(Path('data') / dataset / 'dataset.csv')
    results_dir = Path('output/inwon/with_example')
    results_dir_summary = Path('output/college_confidential_clean/3stage/with_example')

    by_bin = get_by_bin(results_dir, df, bin_range)
    by_bin_summary = get_by_bin(results_dir_summary, df, bin_range)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize = (10,6))

    by_bin['3 Class Correct with Summary'] = by_bin_summary['3 Class Correct']
    by_bin['4 Class Correct with Summary'] = by_bin_summary['4 Class Correct']
    by_bin['3 Class Error Rate'] = 1 - (by_bin['3 Class Correct'] / by_bin['row_count'])
    by_bin['3 Class with Summary Error Rate'] = 1 - (by_bin['3 Class Correct with Summary'] / by_bin['row_count'])
    by_bin['4 Class Error Rate'] = 1 - (by_bin['4 Class Correct'] / by_bin['row_count'])
    by_bin['4 Class with Summary Error Rate'] = 1 - (by_bin['4 Class Correct with Summary'] / by_bin['row_count'])

    by_bin.plot(x="Max Wordcount", y=["3 Class Error Rate", "3 Class with Summary Error Rate"], kind="line",\
                 color=["blue", "red"], ax=ax1) 
    ax1.set_ylabel('3 Class Error Rate')

    by_bin.plot(x="Max Wordcount", y=["4 Class Error Rate", "4 Class with Summary Error Rate"], kind="line",\
                 color=["blue", "red"], ax=ax2) 
    ax2.set_ylabel('4 Class Error Rate')

    for i, tick in enumerate(ax1.get_xticklabels()):
        if i % 5 == 0:
            tick.set_visible(True)
        else:
            # tick.set_visible(False)
            tick.set_visible(True)

    for i, tick in enumerate(ax2.get_xticklabels()):
        if i % 5 == 0:
            tick.set_visible(True)
        else:
            # tick.set_visible(False)  
            tick.set_visible(True)

    fig.suptitle(' '.join([w.capitalize() for w in f'{dataset}_{postfix}'.split('_')]))
    fig.tight_layout()

    # plt.savefig("output.jpg")

    # fig_save_dir = FIGURES_PATH / 'llm' / dataset / postfix
    # fig_save_dir = BASE_PATH / 'plot'
    fig_save_dir = Path('plot')

    fig_save_dir.mkdir(parents=True, exist_ok= True)
    # fig.savefig(fig_save_dir / 'eval_per_token_3stage.pdf', bbox_inches='tight')
    fig.savefig('eval_per_token_3stage_lineformat.pdf', bbox_inches='tight')
    print('save success')

if __name__ == '__main__':
    evaluate()