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
        example_ids += [example.name]
    return examples, example_ids

@click.command()
@click.option('--dataset', default='college_confidential', help='Name of dataset to use')
@click.option('--model', default='upstage-llama2-70b-4bit', help='Name of the model to use')
@click.option('--template', default='inwon', help='Name of template to use for prompts.')
@click.option('--use_example', is_flag=True, help='Use example in prompt')
@click.option('--temperature', default = 1, help='Temperature to use in model')
@click.option('--top_p', default = 0.7, help='Temperature to use in model')
@click.option('--max_tokens', default = 300, help='Maximum number of tokens the model can output')
@click.option('--delay', default = 1, help='Use example in prompt')
@click.option('--max_retry', default = 3, help='Use example in prompt')
def run(
    dataset: str,
    model: str,
    template: str,
    use_example: bool, 
    temperature: float,
    top_p: float,
    max_tokens: int,
    delay: int,
    max_retry: int,
):    
    df = pd.read_csv(DATA_PATH / f'{dataset}/dataset.csv')
    prompt = Prompt.load_template(TEMPLATE_PATH / f'{dataset}/{template}.yaml')
    if use_example:
        examples, example_ids = pick_examples(df)
        prompt.add_examples(examples)
        to_predict = df[~df.index.isin(example_ids)]
    else:
        to_predict = df

    results_dir = OUTPUT_PATH / 'llm' / model / dataset / template
    if use_example:
        results_dir /= 'with_example'
    else:
        results_dir /= 'without_example'
    results_dir.mkdir(parents=True, exist_ok=True)

    with progress_bar() as progress:
        progress_task = progress.add_task(description='Predicting dataset.. ', total = len(to_predict))
        for i, (idx, row) in enumerate(to_predict.iterrows()): 
            text, option_a, option_b, label = row[['text','alternative_a','alternative_b','label']]
            task = Task(text, label, option_a, option_b)
            result_file = results_dir / f'{idx:06d}.json'

            if not result_file.is_file(): 
                print(f'{i}/{len(to_predict)}')
                output,prediction = prompt.execute(
                    task, 
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    delay=delay,
                    max_retry=max_retry,
                )

                if not prediction is None:
                    result = {
                        'index': i,
                        'true_label': label,
                        'model_output': output,
                        'predicted_label': prediction,
                    }
                    json.dump(
                        result,
                        open(result_file, 'w'),
                        indent = 2,
                    )

                print('='*40)
            progress.update(progress_task, advance = 1)

if __name__ == '__main__':
    run()
