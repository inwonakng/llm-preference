from __future__ import annotations
from pathlib import Path
import yaml
import time

from .types import OpenAIMessage, TextGenMessage
from .chat_api import send_request, send_request_openai

DELIMITERS = ['`', '"', '\'', '#', '.', '%']
def remove_delimiters(text: str) -> str:
    for delim in DELIMITERS:
        text = text.replace(delim, '')
    return text.strip()

class Task:
    text: str
    label: int
    alternative_a: str
    alternative_b: str

    def __init__(
        self,
        text: str,
        label: int,
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

    def build_textgen(
        self,
        task:Task,
    ) -> TextGenMessage:
        history = []
        if self.confirmation:
            history += [self.confirmation]
        if self.examples:
            history += self.examples
        return dict(
            user_input = self.wrap_task(task),
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )

    def build_openai(
        self, 
        task: Task,
    ) -> OpenAIMessage:
        messages = [
            {
                'role': 'system',
                'content': self.instruction,
            }
        ]
        if self.confirmation:
            messages += [{
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
    
    def build_retry_textgen(
        self, 
        task: Task, 
        prev_response: str,
    ) -> TextGenMessage:
        history = []
        if self.confirmation:
            history += [self.confirmation]
        if self.examples:
            history += self.examples
        history += [[self.wrap_task(task), prev_response]]
        return dict(
            user_input = self.retry_msg,
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )

    def build_retry_openai(
        self, 
        task: Task, 
        prev_response: str,
    ) -> OpenAIMessage:
        messages = [
            {
                'role': 'system',
                'content': self.instruction,
            } 
        ]
        if self.confirmation:
            messages += [{
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

    def execute(
        self,
        task: Task,
        api_endpoint: str = 'http://localhost:5000/api/v1/chat',
        model: str = 'gpt-4',
        temperature: float = 1,
        top_p: float = 0.7,
        max_tokens: int = 300,
        delay: int = -1,
        max_retry: int = -1,
    ) -> tuple[str, int | None, OpenAIMessage]:
        mode = 'textgen'
        if model in ['gpt-4', 'gpt-3.5-turbo']:
            mode = 'openai'

        textgen_messages = self.build_textgen(task=task)
        openai_messages = self.build_openai(task=task)
        if mode == 'textgen':
            output = send_request(
                api_endpoint, 
                messages = textgen_messages,
                max_tokens = max_tokens,
                temperature = temperature,
                top_p = top_p,
                model_name = model,
            )
        elif mode == 'openai':
            output = send_request_openai(
                model = model, 
                messages = openai_messages,
                max_tokens = max_tokens,
                temperature = temperature,
                top_p = top_p,
            )
        else:
            raise NotImplementedError('Unknown prompt mode')
        prediction = remove_delimiters(output.split('\n')[-1])

        print(task.text)
        print('A:', task.alternative_a)
        print('B:', task.alternative_b)
        print('-'*40)
        print('Model output:')
        print(output)
        print()
        print('True label:', self.label_to_text[task.label])
        print('Model prediction:', prediction)
        if delay > 0:
            time.sleep(3)

        retry_count = 1
        while prediction.lower() not in self.text_to_label:
            if max_retry != -1 and retry_count == max_retry:
                print('Tried enough.. we are aborting')
                return output, None

            textgen_messages = self.build_retry_textgen(
                task=task, 
                prev_response=output,
            )
            openai_messages = self.build_retry_openai(
                task=task, 
                prev_response=output,
            )
            if mode == 'textgen':
                output = send_request(
                    api_endpoint, 
                    messages = textgen_messages,
                    max_tokens = max_tokens,
                    temperature = temperature,
                    top_p = top_p,
                    model_name = model,
                )
            elif mode == 'openai':
                output = send_request_openai(
                    model = model, 
                    messages = openai_messages,
                    max_tokens = max_tokens,
                    temperature = temperature,
                    top_p = top_p,
                )
            else:
                raise NotImplementedError('Unknown prompt mode')
            prediction = remove_delimiters(output.split('\n')[-1])

            print('-'*40)
            print(f'Retry {retry_count} output:')
            print(output)
            print()
            print(f'Prediction:', prediction)
            if delay > 0:
                time.sleep(delay)
            
            retry_count += 1

        return output, self.text_to_label[prediction.lower()], openai_messages
