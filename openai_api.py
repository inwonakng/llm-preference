import os
os.environ['OPENAI_API_KEY']="sk-111111111111111111111111111111111111111111111111"
os.environ['OPENAI_API_BASE']="http://0.0.0.0:5001/v1"
import openai
from tqdm.auto import tqdm
import pandas as pd
import json

from prompt import build_prompt, build_retry_prompt

labels = {
    0: 'A?B',
    1: 'A>B',
    2: 'A<B',
    3: 'A=B',
}

label_to_text = {
    0: 'No preference',
    1: 'A is preferred over B',
    2: 'B is preferred over A',
    3: 'Equal preference',
}

text_to_label = {t.lower(): l for l,t in label_to_text.items()}

df = pd.read_csv('data/CollegeConfidentialComparativeDisucssions/cc_raw_data.csv')
df['Raw_text'] = df['Raw_text'].str.replace('\n',' ')
# Step 1. Pick examples to use.
def pick_examples():
    examples = []
    example_ids = []
    for label, subset in df.groupby('Label'):
        # find the shortest on to use for example
        valid_length = subset[subset['Raw_text'].str.len() > 20]
        example = valid_length.iloc[valid_length['Raw_text'].str.len().argmin()]
        examples += [{
            'comment': example['Raw_text'],
            'option_a': example['Alt_a'],
            'option_b': example['Alt_b'],
            'label': label_to_text[label]
        }]
        example_ids += [example.index]

    return examples, example_ids

examples, example_ids = pick_examples()
results = []

for text, option_a, option_b, label in tqdm(df[~df.index.isin(example_ids)][['Raw_text','Alt_a','Alt_b','Label']].values):
    # while not complete:
    prompt = build_prompt(
        comment = text.replace('\n',''),
            option_a = option_a,
            option_b = option_b,
            examples = examples
    )
    response = openai.ChatCompletion.create(
        model="x",
        messages = prompt
    )
    text = response['choices'][0]['message']['content']
    print('True label:', label_to_text[label])
    print('Model output:', text)

    if text.lower() not in text_to_label:
        success = False
        while not success:
            retry_prompt = build_retry_prompt(prompt, text)
            response = openai.ChatCompletion.create(
                model="x",
                messages = retry_prompt
            )
            text = response['choices'][0]['message']['content']
            # print('retry prompt', retry_prompt)
            print('Retry output:', text)
            success = text.lower() in text_to_label
    print('='*40)

    results += [
        {
            'prompt': prompt,
            'true_label': label,
            'model_output': text,
            'predicted_label': text_to_label[text.lower()],
        }
    ]

results = pd.DataFrame(results)
results.to_csv('openai_api_results.csv', index=False)