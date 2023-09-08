import json

instructions_prompt = [
    {
        'role': 'system',
        'content': '''You are a user on college confidential forums.
            Your job is to detect if there exists a preference given a comment and two options. 
            If there exists a preference, you must detect what the preference is.
            You will be given a comment and two alternatives for each task.
            The options will be denoted by ```Option A:``` and ```Option B:```.
            The comment will be denoted by ```Comment:```.
            If there is no preference, respond with "No preference".
            If option A is preferred over option B, respond with "A is preferred over B".
            If option B is preferred over option A, respond with "B is preferred over A".
            If options A and B are equally preferred, respond with "Equal preference".
            You must respond only using the above four responses. 
        '''
    }
]

retry_outout_prompt = [
    {
        'role': 'user',
        'content': 'You may only respond using the following phrases: "No preference", "A is preferred over B", "B is preferred over A", "Equal preference". Do not use the options\'s real names. Try again.'
    }
]

def build_examples_prompt(examples):
    examples_prompt = []
    for e in examples:
        examples_prompt += [
            {
                'role': 'user',
                'content': f'''```Option A:
                    {e['option_a']}
                    ```

                    ```Option B:
                    {e['option_b']}
                    ```

                    ```Comment:
                    {e['comment']}
                    ```

                    Output:
                '''
            }, {
                'role': 'assistant',
                'content': e['label']
            }
        ]
    return json.loads(' '.join(json.dumps(examples_prompt).split()))

def build_task_prompt(comment, option_a, option_b):
    prompt = [
        {
            'role': 'user',
            'content': f'''```Option A:
                {option_a}
                ```

                ```Option B:
                {option_b}
                ```

                ```Comment: 
                {comment}
                ```
                
                Output:
            '''
        }
    ]
    return json.loads(' '.join(json.dumps(prompt).split()))
    

def build_prompt(comment, option_a, option_b, examples):
    prompt = instructions_prompt + build_examples_prompt(examples) + build_task_prompt(comment, option_a, option_b)
    return json.loads(' '.join(json.dumps(prompt).split()))

def build_retry_prompt(prev_prompt, response):
    return prev_prompt + [{'role': 'assistant', 'content': response}] + retry_outout_prompt