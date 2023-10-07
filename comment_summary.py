import matplotlib.pyplot as plt
import json
import pandas as pd
import click
import numpy as np
from pathlib import Path

@click.command()
@click.option('--dataset', default='college_confidential_clean', help='Name of dataset to use')
@click.option('--template', default='3stage', help='Name of template to use for prompts.')
@click.option('--use_example', is_flag=True, help='Use example in prompt')
def count(
    dataset: str,
    template: str,
    use_example: bool, 
):
    postfix = 'without_example'
    if use_example:
        postfix = 'with_example'

    results_dir = Path('output') / dataset / template / postfix

    results = []    
    for f in results_dir.glob('*.json'):
        result = json.load(open(f))
        if len(result.keys()) == 13: ###############
            summary = result['mean_token_summary_output'] ###############
            comment = result['tokenCount_commet'] ###############
            results += [{
                'summary': summary,
                'comment': comment,
            }]  

    results = pd.DataFrame(results)
    results = results.sort_values(by=['comment'])

    plt.plot(np.array(results['comment']), np.array(results['summary'])) 
    plt.xlabel("Original Comment tokens") 
    plt.ylabel("Summary tokens") 
    
    plt.savefig("OriginalComment_Summary.png") 
    
    print('save success')

if __name__ == '__main__':
    count()