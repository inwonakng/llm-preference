import rich.progress
from pathlib import Path
import json
import click
import yaml
import tiktoken
import numpy as np
import pandas as pd
from prompt import Task, ThreeStagePrompt

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
@click.option('--template', default='3stage', help='Name of template to use for prompts.')
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
    prompt = ThreeStagePrompt.load_template(f'templates/{dataset}/{template}.yaml')

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
    description = f'{dataset} {template} {"with" if use_example else "without"} example {"two stage" if two_stage else "single stage"}'

    ### added by linj26 ### 
    '''load university synonyms'''    
    synonyms = yaml.safe_load(open("./synonyms.json"))
    for i in synonyms.keys():
        synonyms[i] += [i]
    prompt.synonyms = synonyms

    '''load example for summarization'''    
    df_summary = pd.read_csv(f'./example_summary.csv')
    example_summary = []
    for i in range(df_summary.shape[0]):
        example_summary += [[df_summary.iloc[i]["comment_alternative"]] + [df_summary.iloc[i]["response"]]]
    prompt.example_summary = example_summary

    ### added by linj26 ### 
    acc = 0 
    total_tokenCount_summary_input = 0
    total_tokenCount_summary_output = 0
    total_tokenCount_1st_2nd_input = 0
    total_tokenCount_1st_2nd_output = 0

    acc_summary = 0 
    total_tokenCount_summary_input_summary = 0
    total_tokenCount_summary_output_summary = 0
    total_tokenCount_1st_2nd_input_summary = 0
    total_tokenCount_1st_2nd_output_summary = 0
    iter = 0
    encoding = tiktoken.get_encoding("cl100k_base")
    with progress_bar() as progress:
        progress_task = progress.add_task(description=description, total = len(to_predict))
        for i, (idx, row) in enumerate(to_predict.iterrows()): 
            text, option_a, option_b, label = row[['text','alternative_a','alternative_b','label']].values
            task = Task(text, label, option_a, option_b)   
            # print(f'1111111 task.text: {task.text}')

            ############################################################################################################
            ############################## ''' USE summary''' ##########################################################
            ############################################################################################################    

            result_file_withSummary = results_dir / f'{idx:06d}_withSummary.json'
            if not result_file_withSummary.is_file(): 
                length_todo_summar = 1 #100
                success, output, params, token_summary_input, token_summary_output, token_1st_2nd_input,\
                    token_1st_2nd_output = prompt.execute(task, length_todo_summar, verbose = verbose)

                tokenCount_commet = len(encoding.encode(text))
                # print(f'22222222222 task.text: {task.text}')
                # print(f'tokenCount_commet: {tokenCount_commet}')

                tokenCount_summary_input = 0
                tokenCount_summary_output = 0
                tokenCount_1st_2nd_input = 0
                tokenCount_1st_2nd_output = 0
            
                for t in token_summary_input:
                    tokenCount_summary_input += len(encoding.encode(t))
                total_tokenCount_summary_input_summary += tokenCount_summary_input


                mean_token_summary_output = []
                for t in token_summary_output:
                    tokenCount_summary_output += len(encoding.encode(t))
                    mean_token_summary_output.append(tokenCount_summary_output)
                total_tokenCount_summary_output_summary += tokenCount_summary_output
                mean_token_summary_output = (mean_token_summary_output[0] + mean_token_summary_output[-1])//2
                # print(f'tokenCount_summary_output: {tokenCount_summary_output}')
                # print(f'mean_token_summary_output: {mean_token_summary_output}')

                for t in token_1st_2nd_input:
                    tokenCount_1st_2nd_input += len(encoding.encode(t))
                total_tokenCount_1st_2nd_input_summary += tokenCount_1st_2nd_input

                for t in token_1st_2nd_output:
                    tokenCount_1st_2nd_output += len(encoding.encode(t))
                total_tokenCount_1st_2nd_output_summary += tokenCount_1st_2nd_output                

                if not success:
                    print('Failed to generate output!!! Skip.')
                    progress.update(progress_task, advance = 1, description = f'{description} {i}/{len(to_predict)}')
                    continue

                result_summay = {
                    'index': i,
                    'true_label': label,
                    'predicted_label': output,
                    'tokenCount_commet' : tokenCount_commet,
                    'tokenCount_summary_input': tokenCount_summary_input,
                    'tokenCount_summary_output': tokenCount_summary_output,
                    'mean_token_summary_output': mean_token_summary_output,
                    'tokenCount_1st_2nd_input': tokenCount_1st_2nd_input,
                    'tokenCount_1st_2nd_output': tokenCount_1st_2nd_output,
                    'total_tokenCount_summary_input': total_tokenCount_summary_input_summary,
                    'total_tokenCount_summary_output': total_tokenCount_summary_output_summary,   
                    'total_tokenCount_1st_2nd_input': total_tokenCount_1st_2nd_input_summary,   
                    'total_tokenCount_1st_2nd_output': total_tokenCount_1st_2nd_output_summary,   
                }
                json.dump(
                    result_summay,
                    open(result_file_withSummary, 'w'),
                    indent = 2,
                ) 

                if label == output: 
                    acc_summary += 1
                    print('has summary True')
                print(f'i= {i}, acc_withSummary: {acc_summary/(iter+1)}') 


                # if verbose:
                #     # print('final output:', output)
                #     print('='*80)      

            ############################################################################################################
            ############################## ''' No summary''' ###########################################################
            ############################################################################################################ 
            # print('-'*50)  
            result_file = results_dir / f'{idx:06d}_.json'
            if not result_file.is_file(): 
                # print(f'{i}/{len(to_predict)}')
                length_todo_summar = 10**20 #100
                success, output, params, token_summary_input, token_summary_output, token_1st_2nd_input,\
                    token_1st_2nd_output = prompt.execute(task, length_todo_summar, verbose = verbose)

                tokenCount_commet = len(encoding.encode(task.text))
                # print(f'task.text: {task.text}')
                # print(f'tokenCount_commet: {tokenCount_commet}')
                tokenCount_summary_input = 0
                tokenCount_summary_output = 0
                tokenCount_1st_2nd_input = 0
                tokenCount_1st_2nd_output = 0
            
                for t in token_summary_input:
                    tokenCount_summary_input += len(encoding.encode(t))
                total_tokenCount_summary_input += tokenCount_summary_input

                for t in token_summary_output:
                    tokenCount_summary_output += len(encoding.encode(t))
                total_tokenCount_summary_output += tokenCount_summary_output
                # print(f'tokenCount_summary_output: {tokenCount_summary_output}')

                for t in token_1st_2nd_input:
                    tokenCount_1st_2nd_input += len(encoding.encode(t))
                total_tokenCount_1st_2nd_input += tokenCount_1st_2nd_input

                for t in token_1st_2nd_output:
                    tokenCount_1st_2nd_output += len(encoding.encode(t))
                total_tokenCount_1st_2nd_output += tokenCount_1st_2nd_output                

                if not success:
                    print('Failed to generate output!!! Skip.')
                    progress.update(progress_task, advance = 1, description = f'{description} {i}/{len(to_predict)}')
                    continue

                result = {
                    'index': i,
                    'true_label': label,
                    'predicted_label': output,
                    'tokenCount_commet' : tokenCount_commet,
                    'tokenCount_summary_input': tokenCount_summary_input,
                    'tokenCount_summary_output': tokenCount_summary_output,
                    'tokenCount_1st_2nd_input': tokenCount_1st_2nd_input,
                    'tokenCount_1st_2nd_output': tokenCount_1st_2nd_output,
                    'total_tokenCount_summary_input': total_tokenCount_summary_input,
                    'total_tokenCount_summary_output': total_tokenCount_summary_output,   
                    'total_tokenCount_1st_2nd_input': total_tokenCount_1st_2nd_input,   
                    'total_tokenCount_1st_2nd_output': total_tokenCount_1st_2nd_output,   
                }
                json.dump(
                    result,
                    open(result_file, 'w'),
                    indent = 2,
                )
  
                if label == output: 
                    acc += 1
                    print('no summary True')
                print(f'i= {i}, acc: {acc/(iter+1)}')    

                if verbose:
                    # print('final output:', output)
                    print('='*80)       
                               
                iter += 1
            print('-'*50)  
            progress.update(progress_task, advance = 1, description = f'{description} {i}/{len(to_predict)}')
            # if i == 0: break

if __name__ == '__main__':
    run()