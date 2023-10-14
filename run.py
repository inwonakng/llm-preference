import rich.progress
from pathlib import Path
import json
import click
import openai
import time
import pandas as pd
from prompt import Task, Prompt, TwoStagePrompt

from summary import Prompt_summary
import copy

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
@click.option('--model', default='llama2-70b', help='Name of model to use')
@click.option('--template', default='sikai', help='Name of template to use for prompts.')
@click.option('--use_example', is_flag=True, help='Use example in prompt')
@click.option('--two_stage', is_flag=True, help='Use two stage prompt')
@click.option('--mode', default='textgen', help='Mode to use')
@click.option('--verbose', is_flag=True, help='Verbose mode')
@click.option('--do_summary', is_flag=True, help='Do summary')
@click.option('--template_summary', default='summary', help='Name of template to use for summary')
@click.option('--use_example_summary', is_flag=True, help='Use example in summary prompt')
@click.option('--long_text_length', default=200, help='Do summary if text length is longer than long_text_length')

def run(
    dataset: str,
    model: str,
    template: str,
    use_example: bool, 
    two_stage: bool,
    mode: str,
    verbose: bool,
    do_summary: bool,
    template_summary: str,
    use_example_summary: bool,
    long_text_length: int,
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

    if mode == 'openai':
        assert model == 'gpt-4' or model == 'gpt-3.5-turbo'

    results_dir = Path('output') / model / dataset / template
    if use_example:
        results_dir /= 'with_example_' + mode
    else:
        results_dir /= 'without_example_' + mode
    results_dir.mkdir(parents=True, exist_ok=True)

    ################## do summary ################## 
    if do_summary:
        prompt_summar = Prompt_summary.load_template(f"templates/summary/{template_summary}.yaml")
        prompt_summar.load_synonyms()

        if use_example_summary:   
            prompt_summar.add_examples_summary()
    ################## do summary ################## 
    print(f'long_text_length: {long_text_length}')
    # print(prompt)
    description = f'{dataset} {template} {"with" if use_example else "without"}_example {"2-stage" if two_stage else "1-stage"} {mode} {model}'

    iter = 0
    acc = 0
    summary_success = 0
    iter_summary = 0
    with progress_bar() as progress:
        progress_task = progress.add_task(description=description, total = len(to_predict))
        for i, (idx, row) in enumerate(to_predict.iterrows()): 
            text, option_a, option_b, label = row[['text','alternative_a','alternative_b','label']].values
            task = Task(text, label, option_a, option_b)
            result_file = results_dir / f'{idx:06d}.json'
            len_text = len(text.split())

            if do_summary: 
                long_text = long_text_length
            else:
                long_text = float('inf')

            if not result_file.is_file(): 
                if len_text <= long_text:
                    while True:
                        try:
                            # print(f'Getting output for {idx:06d}...')
                            success, output, params = prompt.execute(task, mode = mode, model = model, verbose = verbose)
                        except openai.error.RateLimitError as e:
                            print(f'Rate limit exceeded. Waiting for 5s: {e}')
                            time.sleep(5)
                        except Exception as e:
                            print(f'Some other exception happended: {e}')
                        else:
                            break

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
                ###################################     do summary    ####################################
                else: # len_text > long_text:
                    task_summary = copy.deepcopy(task)
                    summary, retry_cnt = prompt_summar.execute(task_summary, verbose = verbose)
                    task_summary.text = summary

                    while True:
                        try:
                            success, output, params = prompt.execute(task_summary, mode = mode, model = model, verbose = verbose)
                        except openai.error.RateLimitError as e:
                            print(f'Summary part - Rate limit exceeded. Waiting for 5s: {e}')
                            time.sleep(5)
                        except Exception as e:
                            print(f'Summary part - Some other exception happended: {e}')
                        else:
                            break

                    if not success:
                        print('Summary part - Failed to generate output!!! Skip.')
                        progress.update(progress_task, advance = 1, description = f'{description} {i}/{len(to_predict)}')
                        continue

                    if use_example_summary and output == label:
                        prompt_summar.update_examples(task.text, task.alternative_a, task.alternative_b, summary, retry_cnt)

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

                    iter_summary += 1
                    if output == label:
                        summary_success += 1
                        print(f'summary success')

                    print(f'summary_success: {summary_success/iter_summary}')

                iter += 1 
                if output == label: 
                    acc += 1
                print(f'acc {idx:06d}: {acc/iter}')
                print('='*40)
                ##########################################################################################
                if verbose:
                    print('final output:', output)
            progress.update(progress_task, advance = 1, description = f'{description} {i}/{len(to_predict)}')

if __name__ == '__main__':
    run()