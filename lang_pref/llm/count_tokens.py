import click
import tiktoken
import pandas as pd


from .run import pick_examples
from .prompt import Task, Prompt
from lang_pref.config.paths import (
    TEMPLATE_PATH,
    DATA_PATH,
)

price_per_token = {
    'GPT-4-8k': {
        'input': 0.03 / 1e3,
        'output': 0.06 / 1e3,
    }, 
    'GPT-3.5-4k': {
        'input': 0.0015 / 1e3,
        'output': 0.002 / 1e3,
    }, 
    'Claude Instant': {
        'input': 1.63 / 1e6,
        'output': 5.51 / 1e6,
    },
    'Claude 2': {
        'input': 11.02 / 1e6,
        'output': 32.68 / 1e6,
    },
}


def parse_prompt_params(
    prompt_params: dict,
) -> str:
    main_task = prompt_params['user_input'] 
    history = ' '.join([
        user + ' ' + assistant 
        for user,assistant in prompt_params['history']['visible']
    ])
    instruction = prompt_params['context_instruct']
    return main_task + history + instruction

encoding = tiktoken.get_encoding('cl100k_base')
def count_tokens(
    text: str,
    # token_char_count: int = 0,
    # tokens_per_word: float = 2,
) -> int | float:

    # if token_char_count > 0:
    #     token_count = sum([
    #         math.ceil(len(word) / token_char_count)
    #         for word in text.split()
    #     ])
    # else:
    #     token_count = len(text.split()) * tokens_per_word

    return len(encoding.encode(text))

@click.command()
@click.option('--dataset', default='college_confidential', help='Name of dataset to use')
@click.option('--template', default='inwon', help='Name of template to use for prompts.')
@click.option('--use_example', is_flag=True, help='Use example in prompt')
def count_all_tokens(
    dataset: str,
    template: str,
    use_example: bool, 
):
    df = pd.read_csv(DATA_PATH / f'{dataset}/dataset.csv')
    prompt = Prompt.load_template(TEMPLATE_PATH / f'{dataset}/{template}.yaml')
    if use_example:
        examples, example_ids = pick_examples(df)
        prompt.add_examples(examples)
        to_predict = df[~df.index.isin(example_ids)]
    else:
        to_predict = df

    input_tokens_count, output_tokens_count = 0, 0

    for text, option_a, option_b, label in to_predict[['text','alternative_a','alternative_b','label']].values:
        input_tokens_count += count_tokens(
            parse_prompt_params(
                prompt.build(
                    Task(text, label, option_a, option_b)
                )
            )
        )
        output_tokens_count += count_tokens(prompt.label_to_text[label])

    print(f'{len(to_predict):,} tasks')
    print(f'estimated input tokens: {input_tokens_count:,}')
    print(f'estimated output tokens: {output_tokens_count:,}')
    print(f'estimated total tokens: {input_tokens_count + output_tokens_count:,}')
    print('-'*40)

    print()

    for model_name, prices in price_per_token.items():
        print(model_name)
        print(f'Input: ${prices["input"] * input_tokens_count:,.2f}')
        print(f'Output: ${prices["output"] * output_tokens_count:,.2f}')
        print(f'Total: ${prices["input"] * input_tokens_count + prices["output"] * output_tokens_count:,.2f}')
        print('='*40)

if __name__ == '__main__':
    count_all_tokens()
