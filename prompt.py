import json

instructions_prompt = [
    {
        'role': 'system',
        'content': '''Pretend that you are a user on college confidential forums.
            Your job is to detect if there exists a preference between two options in a comment. 
            If there exists a preference, you must detect what the preference is.
            If the author of the comment expresses an explicit preference, you must detect it.
            You will be given a comment and two alternatives for each task.
            The options will be denoted by ```Option A:``` and ```Option B:```.
            The comment will be denoted by ```Comment:```.
            
            Rules:
            - You MUST NOT respond with a summary of the comment.
            - You MUST NOT use the options' real names.
            - You MUST refer to the options as A or B. 
            - You MUST respond with ```No preference``` if there is no strict preference.
            - You MUST respond with ```A is preferred over B``` if option A is preferred over option B.
            - You MUST respond with ```B is preferred over A``` if option B is preferred over option A.
            - You MUST respond with ```Equal preference``` if options A and B are equally preferred.
            - You MUST respond using one of the four phrases above. 
        '''
    }
]

confirmation_prompt = [
    {
        'role': 'user',
        'content': 'Do you understand the rules and your job? Repeat your role, job and the rules.'
    },{
        'role': 'assistant',
        'content': '''I am a user on college confidential.
        My job is to determine the preference over different options in a comment.
        Here are the rules of my job: 
        - I must only respond with: "No preference", "A is preferred over B", "B is preferred over A" and "Equal preference".
        - I must respond with "No preference" if there is no strict preference.
        - I must respond with "A is preferred over B" if option A is preferred over option B.
        - I must respond with "B is preferred over A" if option B is preferred over option A.
        - I must respond with "Equal preference" if options A and B are equally preferred.
        - I must not respond with any other response.
        '''
    }
]

retry_outout_prompt = [
    {
        'role': 'user',
        'content': '''Your response was incorrect. 
        Let's try again.
        Here is a reminder of the rules:

        - You MUST NOT respond with any other details than the preference expressed in the comment.
        - You MUST only report the preference in the comment.
        - You MUST respond only using one of the following phrases: ```No preference```, ```A is preferred over B```, ```B is preferred over A```, ```Equal preference```. Do not say anything else.
        - You MUST NOT use the options's real names.
        - You MUST only refer to the options as A or B.
        - You MUST NOT explain your reasoning, only respond with the given phrase.

        Now try again and respond with a correct response to the previous comment.
        '''
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
            '''
        }
    ]
    return json.loads(' '.join(json.dumps(prompt).split()))
    

def build_prompt(comment, option_a, option_b, examples):
    prompt = instructions_prompt + build_examples_prompt(examples) + build_task_prompt(comment, option_a, option_b)
    return json.loads(' '.join(json.dumps(prompt).split()))

def build_retry_prompt(prev_prompt, response):
    return prev_prompt + [{'role': 'assistant', 'content': response}] + retry_outout_prompt