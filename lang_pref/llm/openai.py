import json
import click

import pandas as pd
from .prompt import Task, Prompt
from lang_pref.utils.progress import progress_bar
from lang_pref.config.paths import (
    OUTPUT_PATH,
    TEMPLATE_PATH,
    DATA_PATH,
)

@click.command()
@click.option('--dataset', default='college_confidential', help='Name of dataset to use')
@click.option('--model', default='gpt-4', help='Name of the model to use')
@click.option('--template', default='inwon', help='Name of template to use for prompts.')
def run_openai(
    dataset: str,
    model: str,
    template: str,
):
    df = pd.read_csv(DATA_PATH / f'{dataset}/dataset.csv')
    prompt = Prompt.load_template(TEMPLATE_PATH / f'{dataset}/{template}.yaml')
    to_predict = df

    results_dir = OUTPUT_PATH / 'llm' / model / dataset / template
    results_dir /= 'without_example'
    results_dir.mkdir(parents=True, exist_ok=True)

    with progress_bar() as progress:
        progress_task = progress.add_task(description='Predicting dataset.. ', total = len(to_predict))
        for i, (idx, row) in enumerate(to_predict.iterrows()): 
            text, option_a, option_b, label = row[['text','alternative_a','alternative_b','label']].values
            task = Task(text, label, option_a, option_b)
            result_file = results_dir / f'{idx:06d}.json'

            if not result_file.is_file(): 
                print(f'{i}/{len(to_predict)}')
                output = prompt.execute(
                    task, 
                    mode='openai', 
                    model='gpt-4',
                    max_retry=3,
                )
                if not output is None:
                    result = {
                        'index': i,
                        'true_label': label,
                        'predicted_label': output,
                    }
                    json.dump(
                        result,
                        open(result_file, 'w'),
                        indent = 2,
                    )
                print('='*40)
            progress.update(progress_task, advance = 1)

if __name__ == '__main__':
    run_openai()
