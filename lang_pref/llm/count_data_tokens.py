import click
import tiktoken
import pandas as pd
import numpy as np

from lang_pref.config.paths import (
    DATA_PATH,
)

encoding = tiktoken.get_encoding('cl100k_base')
def count_tokens(
    text: str,
) -> int | float:
    return len(encoding.encode(text))

@click.command()
@click.option('--dataset', default='college_confidential', help='Name of dataset to use')
def count_all_tokens(
    dataset: str,
):
    df = pd.read_csv(DATA_PATH / f'{dataset}/dataset.csv')

    input_tokens_count = []
    for text in df['text'].values:
        # print(text[0])
        # print(len(text))
        input_tokens_count += [count_tokens(text)]

    input_tokens_count = np.array(input_tokens_count)
    print('Dataset tokens')
    print('Mean:', input_tokens_count.mean())
    print('Std:', input_tokens_count.std())
    print('Min:', input_tokens_count.min())
    print('Max:', input_tokens_count.max())

if __name__ == '__main__':
    count_all_tokens()
