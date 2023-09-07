import json

instructions_prompt = [
    {
        'role': 'system',
        'content': '''You are a user on college confidential forums.
            Your job is to detect if there exists a preference between the two options in the comment, and if it does, what the preference is.
            You will be given a comment and two alternatives for each task.
            The comment will be denoted by ```Comment:```.
            The options will be denoted by ```Option A:``` and ```Option B:```.
            If there is no preference, only respond with "No preference".
            If option A is preferred over option B, only respond with "A is preferred over B".
            If option B is preferred over option A, only respond with "B is preferred over A".
            If options A and B are equally preferred, only respond with "Equal preference".
            You may only respond using the above four responses. 
        '''
    }
]

clean_outout_prompt = [
    {
        'role': 'user',
        'content': 'You may only respond using the following phrases: "No preference", "A is preferred over B", "B is preferred over A", "Equal preference". Try again.'
    }
]

def build_examples_prompt(examples):
    examples_prompt = []
    for e in examples:
        examples_prompt += [
            {
                'role': 'user',
                'content': f'''
                    ```Comment:
                    {e['comment']}
                    ```

                    ```Option A:
                    {e['option_a']}
                    ```

                    ```Option B:
                    {e['option_b']}
                    ```

                    Output:
                '''
            }, {
                'role': 'assistant',
                'content': e['label']
            }
        ]
    return examples_prompt

def build_task_prompt(comment, option_a, option_b):
    return [
        {
            'role': 'user',
            'content': f'''
                ```Comment: 
                {comment}
                ```

                ```Option A:
                {option_a}
                ```

                ```Option B:
                {option_b}
                ```

                Output:
            '''
        }
    ]
    

def build_prompt(comment, option_a, option_b, examples):
    prompt = instructions_prompt + build_examples_prompt(examples) + build_task_prompt(comment, option_a, option_b)
    return json.loads(' '.join(json.dumps(prompt).split()))

def build_retry_prompt(prev_prompt, response):
    return prev_prompt + [{'role': 'assistant', 'content': response}] + clean_outout_prompt