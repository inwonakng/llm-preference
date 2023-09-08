import pandas as pd
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import confusion_matrix, get_scorer

from utils import chat_api, progress_bar
from prompt import (
    instructions_prompt,
    build_task_prompt,
    build_examples_prompt,
    retry_outout_prompt,
    confirmation_prompt
)

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

metrics = ['accuracy', 'balanced_accuracy', 'f1_micro']

text_to_label = {t.lower(): l for l,t in label_to_text.items()}

df = pd.read_csv('data/CollegeConfidentialComparativeDisucssions/cc_raw_data.csv')
df['Raw_text'] = df['Raw_text'].str.replace('\n',' ')
# Step 1. Pick examples to use.
def pick_examples():
    examples = []
    example_ids = []
    for label, subset in df.groupby('Label'):
        # find the shortest on to use for example
        valid_length = subset[subset['Raw_text'].str.len() > 100]
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
# print(examples)
# quit()

chat_history = [
    [p['content'] for p in confirmation_prompt]
] + [
    [p['content'] for p in build_examples_prompt([e])] 
    for e in examples
]

# print(chat_history)
# quit()
results_dir = Path('output')
results_dir.mkdir(parents=True, exist_ok=True)

to_predict = df[~df.index.isin(example_ids)]

true_labels = to_predict['Label'].values
predicted_labels = np.ones_like(true_labels) * -1

with progress_bar() as progress:
    progress_task = progress.add_task(description='Predicting dataset.. ', total = len(to_predict))
    for i, (idx, row) in enumerate(to_predict.iterrows()): 
        text, option_a, option_b, label = row[['Raw_text','Alt_a','Alt_b','Label']].values
        result_file = results_dir / f'{idx:06d}.json'

        if not result_file.is_file(): 
            task = build_task_prompt(
                comment = text.replace('\n',''),
                option_a = option_a,
                option_b = option_b,
            )[0]['content']
            output = chat_api(
                context= instructions_prompt[0]['content'],
                task = task,
                history = {
                    'internal': chat_history,
                    'visible': chat_history,
                },
            )
            print(f'{i}/{len(to_predict)}')
            print(text)
            print('A:', option_a)
            print('B:', option_b)
            print('True label:', label_to_text[label])
            print('Model output:', output)

            if output.lower() not in text_to_label:
                success = False
                while not success:
                    retry_history = chat_history + [[task, output]]
                    output = chat_api(
                        context= instructions_prompt[0]['content'],
                        task = retry_outout_prompt[0]['content'],
                        history = {
                            'internal': retry_history,
                            'visible': retry_history,
                        },
                    )
                    print('Retry output:', output)
                    success = output.lower() in text_to_label
            result = {
                'index': i,
                'true_label': label,
                'model_output': output,
                'predicted_label': text_to_label[output.lower()],
            }
            json.dump(
                result,
                open(result_file, 'w'),
                indent = 2,
            )
            print('='*40)
        # else:
        #     result = json.load(open(result_file))


        progress.update(progress_task, advance = 1)
        # break