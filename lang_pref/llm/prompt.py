from __future__ import annotations
from typing import Literal
from pathlib import Path
import yaml
import time
import os
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
import openai
openai.api_key = OPENAI_API_KEY

from .chat_api import DEFAULT_CHAT_PARAMS, send_request

DELIMITERS = ['```', '"""']
def remove_delimiters(text: str) -> str:
    for delim in DELIMITERS:
        text = text.replace(delim, '')
    return text

class Task:
    text: str
    label: str
    alternative_a: str
    alternative_b: str

    def __init__(
        self,
        text: str,
        label: str,
        alternative_a: str,
        alternative_b: str,
    ) -> None:
        self.text = text
        self.label = label
        self.alternative_a = alternative_a
        self.alternative_b = alternative_b


class Prompt:
    instruction: str
    retry_msg: str
    task_template: str
    label_to_text: dict[int, str]
    text_to_label: dict[str, int]
    confirmation: list[str] | None = None
    examples: list[list[str]] = []

    def __init__(
        self,
        instruction: str,
        retry_msg: str,
        task_template: str,
        label_to_text: dict[int, str],
    ) -> None:
        self.instruction = instruction
        self.retry_msg = retry_msg
        self.task_template = task_template
        self.label_to_text = label_to_text
        self.text_to_label = {
            text.lower(): label 
            for label,text in label_to_text.items()
        }

    @staticmethod
    def load_template(template_path: str | Path) -> Prompt:
        config = yaml.safe_load(open(template_path))
        prompt = Prompt(
            instruction = config['instruction'],
            retry_msg = config['retry_msg'],
            task_template = config['task'],
            label_to_text = config['label'],
        )
        if 'confirmation' in config:
            prompt.add_confirmation(config['confirmation'])
        return prompt

    def add_confirmation(self, confirmation: list[str]):
        self.confirmation = confirmation

    def wrap_task(self, task: Task) -> str:
        return self.task_template.replace(
            '{alternative_a}', task.alternative_a
        ).replace(
            '{alternative_b}', task.alternative_b
        ).replace(
            '{text}', task.text
        )
        
    def add_examples(
        self,
        examples: list[Task]
    ) -> None:
        self.examples = [
            [
                self.wrap_task(e),
                self.label_to_text[e.label]
            ]
            for e in examples
        ]

    def build(
        self, 
        task: Task,
        mode: Literal['openai', 'textgen'] = 'textgen'
    ) -> dict[str, any] | list[dict[str, str]]:
        if mode == 'textgen':
            history = [self.confirmation]
            if self.examples:
                history += self.examples
            return dict(
                **DEFAULT_CHAT_PARAMS,
                user_input = self.wrap_task(task),
                history = dict(
                    internal = history,
                    visible = history
                ),
                context_instruct = self.instruction,
            )
        elif mode == 'openai':
            messages = [
                {
                    'role': 'system',
                    'content': self.instruction,
                },{
                    'role': 'user',
                    'content': self.confirmation[0],
                },{
                    'role': 'assistant',
                    'content': self.confirmation[1],
                }
            ]
            if self.examples:
                for example in self.examples:
                    messages += [
                        {
                            'role': 'user',
                            'content': example[0] 
                        },{
                            'role': 'user',
                            'content': example[1]

                        }
                    ]
            messages += [{
                'role': 'user',
                'content': self.wrap_task(task)
            }]
            return messages
        else:
            raise NotImplementedError('Unknown prompt mode')
    
    def build_retry(
        self, 
        task: Task, 
        prev_response: str,
        mode: Literal['openai', 'textgen'] = 'textgen'
    ) -> dict[str, any]:
        if mode == 'textgen':
            history = [self.confirmation]
            if self.examples:
                history += self.examples
            history += [[self.wrap_task(task), prev_response]]
            return dict(
                **DEFAULT_CHAT_PARAMS,
                user_input = self.retry_msg,
                history = dict(
                    internal = history,
                    visible = history
                ),
                context_instruct = self.instruction,
            )
        elif mode == 'openai':
            messages = [
                {
                    'role': 'system',
                    'content': self.instruction,
                },{
                    'role': 'user',
                    'content': self.confirmation[0],
                },{
                    'role': 'assistant',
                    'content': self.confirmation[1],
                }
            ]
            if self.examples:
                for example in self.examples:
                    messages += [
                        {
                            'role': 'user',
                            'content': example[0] 
                        },{
                            'role': 'user',
                            'content': example[1]

                        }
                    ]
            messages += [
                {
                    'role': 'user',
                    'content': self.wrap_task(task)
                },{
                    'role': 'assistant',
                    'content': prev_response
                },{
                    'role': 'user',
                    'content': self.retry_msg
                }
            ]
            return messages

        else:
            raise NotImplementedError('Unknown prompt mode')


    def execute(
        self,
        task: Task,
        api_endpoint: str = 'http://localhost:5000/api/v1/chat',
        model: str = 'gpt-4',
        delay: int = -1,
        max_retry: int = -1,
    ) -> int | None:
        params = self.build(task=task, mode=mode)

        mode = 'textgen'
        if model in ['gpt-4']:
            mode = 'openai'

        if mode == 'textgen':
            output = send_request(api_endpoint, params)
        elif mode == 'openai':
            output = openai.ChatCompletion.create(
                model = model, 
                max_tokens = 10,
                messages = params,
                temperature = 1.2,
            )['choices'][0]['message']['content']
        else:
            raise NotImplementedError('Unknown prompt mode')
        output = remove_delimiters(output)

        print(task.text)
        print('A:', task.alternative_a)
        print('B:', task.alternative_b)
        print('True label:', self.label_to_text[task.label])
        print('Model output:', output)
        if delay > 0:
            time.sleep(3)

        retry_count = 1
        while output.lower() not in self.text_to_label:
            params = self.build_retry(
                task=task, 
                prev_response=output,
                mode=mode
            )
            if mode == 'textgen':
                output = send_request(api_endpoint, params).replace()
            elif mode == 'openai':
                output = openai.ChatCompletion.create(
                    model = model, 
                    max_tokens = 10,
                    messages = params,
                    temperature = 1.2,
                )['choices'][0]['message']['content']
            else:
                raise NotImplementedError('Unknown prompt mode')
            output = remove_delimiters(output)

            print(f'Retry {retry_count} output:', output)
            if delay > 0:
                time.sleep(delay)

            if max_retry != -1 and retry_count == max_retry:
                print('Tried enough.. we are aborting')
                return None
            
            retry_count += 1
        
        return self.text_to_label[output.lower()]

