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
        # if len(result.keys()) == len_result: 
        idx = int(f.stem)
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
    short_few_shot = Path('/home/linj26/llm/llm-preference/output/llama2-70b/college_confidential_clean/sikai2/with_example_textgen')
    short_zero_shot = Path('/home/linj26/llm/llm-preference/output/llama2-70b/college_confidential_clean/sikai2/without_example_textgen')
    long_few_shot = Path('/home/linj26/llm/llm-preference_Inwon/output/llm/upstage-llama2-70b-4bit/college_confidential/inwon/with_example')
    long_zero_shot = Path('/home/linj26/llm/llm-preference_Inwon/output/llm/upstage-llama2-70b-4bit/college_confidential/inwon/without_example')

    short_few_shot = get_by_bin(short_few_shot, df, bin_range)
    short_zero_shot = get_by_bin(short_zero_shot, df, bin_range)
    long_few_shot = get_by_bin(long_few_shot, df, bin_range)
    long_zero_shot = get_by_bin(long_zero_shot, df, bin_range)

    by_bin = pd.DataFrame()
    by_bin['Max Wordcount'] = short_few_shot['Max Wordcount']
    by_bin['row_count'] = short_few_shot['row_count']
    prompt = [short_few_shot, short_zero_shot, long_few_shot, long_zero_shot]
    prompt_name = ['short_few_shot', 'short_zero_shot', 'long_few_shot', 'long_zero_shot']
    class_name = ['3 Class Correct', '4 Class Correct']
    for i in class_name:
        for j in range(len(prompt_name)):
            by_bin[f"{i} {prompt_name[j]}"] = prompt[j][i] 

    for i in by_bin.columns:
        if i == 'Max Wordcount' or i == 'row_count': continue
        by_bin[f"{i} Error Rate"] = 1 - (by_bin[i] / by_bin['row_count'])

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize = (10,6))

    class3 = []
    class4 = []
    for i in by_bin.columns:
        if i.startswith('3') and i.endswith('Rate'):
            class3.append(i)
        if i.startswith('4') and i.endswith('Rate'):
            class4.append(i)            

    by_bin.plot(x="Max Wordcount", y=class3, kind="line", color=["blue", "red", "yellow", "green"], ax=ax1) 
    ax1.set_ylabel('3 Class Error Rate')

    by_bin.plot(x="Max Wordcount", y=class4, kind="line", color=["blue", "red", "yellow", "green"], ax=ax2) 
    ax2.set_ylabel('4 Class Error Rate')

    for i, tick in enumerate(ax1.get_xticklabels()):
        if i % 5 == 0:
            tick.set_visible(True)
        else:
            tick.set_visible(True)

    for i, tick in enumerate(ax2.get_xticklabels()):
        if i % 5 == 0:
            tick.set_visible(True)
        else:
            tick.set_visible(True)

    fig.suptitle(' '.join([w.capitalize() for w in f'{dataset}_{postfix}'.split('_')]))
    fig.tight_layout()

    save_name = f"output_{'sikai2'}_test"
    # plt.savefig(save_name + ".jpg")

    # fig_save_dir = FIGURES_PATH / 'llm' / dataset / postfix
    # fig_save_dir = BASE_PATH / 'plot'
    fig_save_dir = Path('plot')

    fig_save_dir.mkdir(parents=True, exist_ok= True)
    # fig.savefig(fig_save_dir / 'eval_per_token_3stage.pdf', bbox_inches='tight')
    # fig.savefig('eval_per_token_lose_rate.pdf', bbox_inches='tight')
    print('save success')

if __name__ == '__main__':
    evaluate()