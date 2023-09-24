import rich.progress
from pathlib import Path
import json
import click

import pandas as pd
from prompt import Task, Prompt, TwoStagePrompt

def pick_examples(dataset):
    examples, example_ids = [], []
    for label, subset in dataset.groupby('label'):
        # find the shortest on to use for example
        valid_length = subset[subset['text'].str.len() > 100]
        example = valid_length.iloc[valid_length['text'].str.len().argmin()]
        examples += [
            Task(
                text = example['text'], 
                alternative_a = example['alternative_a'],
                alternative_b = example['alternative_b'],
                label = label
            )
        ]
        example_ids += [example.index]
    return examples, example_ids

def progress_bar():
    return rich.progress.Progress(
        '[progress.description]{task.description}',
        rich.progress.BarColumn(),
        '[progress.percentage]{task.percentage:>3.0f}%',
        rich.progress.TimeRemainingColumn(),
        rich.progress.TimeElapsedColumn(),
        transient=True,
    )

@click.command()
@click.option('--dataset', default='college_confidential_clean', help='Name of dataset to use')
@click.option('--template', default='sikai', help='Name of template to use for prompts.')
@click.option('--use_example', is_flag=True, help='Use example in prompt')
@click.option('--two_stage', is_flag=True, help='Use two stage prompt')
@click.option('--verbose', is_flag=True, help='Verbose mode')
def run(
    dataset: str,
    template: str,
    use_example: bool, 
    two_stage: bool,
    verbose: bool,
):    
    df = pd.read_csv(f'data/{dataset}/dataset.csv')
    if not two_stage:
        prompt = Prompt.load_template(f'templates/{dataset}/{template}.yaml')
    else:
        prompt = TwoStagePrompt.load_template(f'templates/{dataset}/{template}.yaml')

    if use_example:
        examples, example_ids = pick_examples(df)
        prompt.add_examples(examples)
        to_predict = df[~df.index.isin(example_ids)]
    else:
        to_predict = df

    results_dir = Path('output') / dataset / template
    if use_example:
        results_dir /= 'with_example'
    else:
        results_dir /= 'without_example'
    results_dir.mkdir(parents=True, exist_ok=True)

    print(prompt)
    description = f'{dataset} {template} {"with" if use_example else "without"} example {"two stage" if two_stage else "single stage"}'

    with progress_bar() as progress:
        progress_task = progress.add_task(description=description, total = len(to_predict))
        for i, (idx, row) in enumerate(to_predict.iterrows()): 
            text, option_a, option_b, label = row[['text','alternative_a','alternative_b','label']].values
            task = Task(text, label, option_a, option_b)
            result_file = results_dir / f'{idx:06d}.json'

            if not result_file.is_file(): 
                # print(f'{i}/{len(to_predict)}')
                success, output, params = prompt.execute(task, verbose = verbose)

                if not success:
                    print('Failed to generate output!!! Skip.')
                    progress.update(progress_task, advance = 1, description = f'{description} {i}/{len(to_predict)}')
                    continue

                result = {
                    'index': i,
                    'true_label': label,
                    'predicted_label': output,
                    'params': params,
                }
                json.dump(
                    result,
                    open(result_file, 'w'),
                    indent = 2,
                )
                if verbose:
                    print('final output:', output)
                    print('='*40)
            progress.update(progress_task, advance = 1, description = f'{description} {i}/{len(to_predict)}')

if __name__ == '__main__':
    run()