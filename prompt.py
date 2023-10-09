from __future__ import annotations
import yaml
import os
import openai
from typing import Literal

from chat_api import DEFAULT_CHAT_PARAMS, send_request

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

DELIMITERS = ['`', '"', '\'', '#', '.', '%']
def remove_delimiters(text: str) -> str:
    for delim in DELIMITERS:
        text = text.replace(delim, '')
    return text.strip()

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
    examples: list
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
    def load_template(template_path: str) -> Prompt:
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

    def build(self, task: Task, mode: Literal['openai', 'textgen'] = 'textgen') -> dict[str, any] | list[dict[str, str]]:
        if mode == 'textgen':
            if self.confirmation is not None:
                history = [self.confirmation]
            else:
                history = []
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
            messages = [{
                'role': 'system',
                'content': self.instruction,
            }]
            if self.confirmation is not None:
                messages += [{
                    'role': 'user',
                    'content': self.confirmation[0],
                }, {
                    'role': 'assistant',
                    'content': self.confirmation[1],
                }]

            if self.examples:
                for example in self.examples:
                    messages += [{
                        'role': 'user',
                        'content': example[0] 
                    }, {
                        'role': 'assistant',
                        'content': example[1]
                    }]

            messages += [{
                'role': 'user',
                'content': self.wrap_task(task)
            }]
            return messages
        else:
            raise NotImplementedError(f'Invalid mode {mode}')
    
    def build_retry(self, task: Task, prev_response: str, mode: Literal['openai', 'textgen'] = 'textgen') -> dict[str, any] | list[dict[str, str]]:
        if mode == 'textgen':
            if self.confirmation is not None:
                history = [self.confirmation]
            else:
                history = []
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
            messages = [{
                'role': 'system',
                'content': self.instruction,
            }]
            if self.confirmation is not None:
                messages += [{
                    'role': 'user',
                    'content': self.confirmation[0],
                }, {
                    'role': 'assistant',
                    'content': self.confirmation[1],
                }]

            if self.examples:
                for example in self.examples:
                    messages += [{
                        'role': 'user',
                        'content': example[0] 
                    }, {
                        'role': 'assistant',
                        'content': example[1]
                    }]

            messages += [{
                'role': 'user',
                'content': self.wrap_task(task)
            }, {
                'role': 'assistant',
                'content': prev_response
            }, {
                'role': 'user',
                'content': self.retry_msg
            }]
            return messages
        else:
            raise NotImplementedError(f'Invalid mode {mode}')    
    
    def execute(
        self,
        task: Task,
        api_endpoint: str = 'http://localhost:5000/api/v1/chat',
        mode: Literal['openai', 'textgen'] = 'textgen',
        model: str = 'gpt-4',
        verbose: bool = False,
    ):
        if mode == 'openai':
            assert model == 'gpt-4' or model == 'gpt-3.5-turbo'

        params = self.build(task, mode)
        if mode == 'textgen':
            output = send_request(api_endpoint, params)
            if verbose:
                print('Comment:', task.text)
                print('A:', task.alternative_a)
                print('B:', task.alternative_b)
                print('True label:', self.label_to_text[task.label])
                print('Model output:', output)
            output = remove_delimiters(output)

            retry_cnt = 0
            while output.lower() not in self.text_to_label:
                retry_cnt += 1
                params = self.build_retry(task, output)
                output = send_request(api_endpoint, params)
                if verbose:
                    print('Retry output:', output)
                output = remove_delimiters(output)
                if retry_cnt == 10:
                    return False, None, params
            
            return True, self.text_to_label[output.lower()], params
        elif mode == 'openai':
            output = openai.ChatCompletion.create(
                model = model, 
                max_tokens = 10,
                messages = params,
                temperature = 1.,
                top_p = .7,
            )['choices'][0]['message']['content']
            if verbose:
                print('Comment:', task.text)
                print('A:', task.alternative_a)
                print('B:', task.alternative_b)
                print('True label:', self.label_to_text[task.label])
                print('Model output:', output)
            output = remove_delimiters(output)

            retry_cnt = 0
            while output.lower() not in self.text_to_label:
                retry_cnt += 1
                params = self.build_retry(task, output, mode)
                output = openai.ChatCompletion.create(
                    model = 'gpt-4', 
                    max_tokens = 10,
                    messages = params,
                    temperature = 1.,
                    top_p = .7,
                )['choices'][0]['message']['content']
                if verbose:
                    print('Retry output:', output)
                output = remove_delimiters(output)
                if retry_cnt == 10:
                    return False, None, params
            
            return True, self.text_to_label[output.lower()], params
        else:
            raise NotImplementedError(f'Invalid mode {mode}')

    def __str__(self) -> str:
        return f'instruction:\n{self.instruction}\nretry_msg:\n{self.retry_msg}\ntask_template:\n{self.task_template}\nlabel_to_text:\n{self.label_to_text}\nconfirmation:\n{self.confirmation}\nexamples:\n{self.examples}'
    

class TwoStagePrompt:
    instruction: str
    retry_msg_1st_stage: str
    retry_msg_2nd_stage: str
    examples: list
    task_template: str
    label_to_text_1st_stage: dict[int, str]
    text_to_label_1st_stage: dict[str, int]
    label_to_text_2nd_stage: dict[int, str]
    text_to_label_2nd_stage: dict[str, int]
    confirmation: list[str] | None = None
    question_1st_stage: str
    question_2nd_stage: str
    examples: list[list[str]] = []

    def __init__(
        self,
        instruction: str,
        retry_msg_1st_stage: str,
        retry_msg_2nd_stage: str,
        task_template: str,
        question_1st_stage: str,
        question_2nd_stage: str,
        label_to_text: dict[int, str],
        label_binary: dict[int, str],
    ) -> None:
        self.instruction = instruction
        self.retry_msg_1st_stage = retry_msg_1st_stage
        self.retry_msg_2nd_stage = retry_msg_2nd_stage
        self.task_template = task_template
        self.question_1st_stage = question_1st_stage
        self.question_2nd_stage = question_2nd_stage
        self.label_to_text_1st_stage = label_binary
        self.text_to_label_1st_stage = {
            text.lower(): label 
            for label,text in label_binary.items()
        }
        self.label_to_text_2nd_stage = label_to_text
        self.text_to_label_2nd_stage = {
            text.lower(): label 
            for label,text in label_to_text.items()
        }

    @staticmethod
    def load_template(template_path: str) -> Prompt:
        config = yaml.safe_load(open(template_path))
        prompt = TwoStagePrompt(
            instruction = config['instruction'],
            retry_msg_1st_stage = config['retry_msg_1st_stage'],
            retry_msg_2nd_stage = config['retry_msg_2nd_stage'],
            task_template = config['task'],
            question_1st_stage = config['question_1st_stage'],
            question_2nd_stage = config['question_2nd_stage'],
            label_to_text = config['label'],
            label_binary = config['label_binary'],
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
        self.examples = []
        for e in examples:
            if e.label == 0:
                self.examples.append([self.wrap_task(e) + self.question_1st_stage, self.label_to_text_1st_stage[0]])
            else:
                self.examples.append([self.wrap_task(e) + self.question_1st_stage, self.label_to_text_1st_stage[1]])
                self.examples.append([self.question_2nd_stage, self.label_to_text_2nd_stage[e.label]])
    
    def build_two_stage_1(self, task: Task, mode: Literal['openai', 'textgen'] = 'textgen') -> dict[str, any] | list[dict[str, str]]:
        if mode == 'textgen':
            if self.confirmation is not None:
                history = [self.confirmation]
            else:
                history = []
            if self.examples:
                history += self.examples
            return dict(
                **DEFAULT_CHAT_PARAMS,
                user_input = self.wrap_task(task) + self.question_1st_stage,
                history = dict(
                    internal = history,
                    visible = history
                ),
                context_instruct = self.instruction,
            )
        elif mode == 'openai':
            messages = [{
                'role': 'system',
                'content': self.instruction,
            }]
            if self.confirmation is not None:
                messages += [{
                    'role': 'user',
                    'content': self.confirmation[0],
                }, {
                    'role': 'assistant',
                    'content': self.confirmation[1],
                }]

            if self.examples:
                for example in self.examples:
                    messages += [{
                        'role': 'user',
                        'content': example[0] 
                    }, {
                        'role': 'assistant',
                        'content': example[1]
                    }]

            messages += [{
                'role': 'user',
                'content': self.wrap_task(task) + self.question_1st_stage
            }]
            return messages
        else:
            raise NotImplementedError(f'Invalid mode {mode}')
    
    def build_retry_two_stage_1(self, task: Task, prev_response: str, mode: Literal['openai', 'textgen'] = 'textgen') -> dict[str, any] | list[dict[str, str]]:
        if mode == 'textgen':
            if self.confirmation is not None:
                history = [self.confirmation]
            else:
                history = []
            if self.examples:
                history += self.examples
            history += [[self.wrap_task(task) + self.question_1st_stage, prev_response]]
            return dict(
                **DEFAULT_CHAT_PARAMS,
                user_input = self.retry_msg_1st_stage,
                history = dict(
                    internal = history,
                    visible = history
                ),
                context_instruct = self.instruction,
            )
        elif mode == 'openai':
            messages = [{
                'role': 'system',
                'content': self.instruction,
            }]
            if self.confirmation is not None:
                messages += [{
                    'role': 'user',
                    'content': self.confirmation[0],
                }, {
                    'role': 'assistant',
                    'content': self.confirmation[1],
                }]

            if self.examples:
                for example in self.examples:
                    messages += [{
                        'role': 'user',
                        'content': example[0] 
                    }, {
                        'role': 'assistant',
                        'content': example[1]
                    }]

            messages += [{
                'role': 'user',
                'content': self.wrap_task(task) + self.question_1st_stage
            }, {
                'role': 'assistant',
                'content': prev_response
            }, {
                'role': 'user',
                'content': self.retry_msg_1st_stage
            }]
            return messages
        else:
            raise NotImplementedError(f'Invalid mode {mode}')

    def build_two_stage_2(self, task: Task, mode: Literal['openai', 'textgen'] = 'textgen') -> dict[str, any] | list[dict[str, str]]:
        if mode == 'textgen':
            if self.confirmation is not None:
                history = [self.confirmation]
            else:
                history = []
            if self.examples:
                history += self.examples
            history += [[self.wrap_task(task) + self.question_1st_stage, self.label_to_text_1st_stage[1]]]
            return dict(
                **DEFAULT_CHAT_PARAMS,
                user_input = self.question_2nd_stage,
                history = dict(
                    internal = history,
                    visible = history
                ),
                context_instruct = self.instruction,
            )
        elif mode == 'openai':
            messages = [{
                'role': 'system',
                'content': self.instruction,
            }]
            if self.confirmation is not None:
                messages += [{
                    'role': 'user',
                    'content': self.confirmation[0],
                }, {
                    'role': 'assistant',
                    'content': self.confirmation[1],
                }]

            if self.examples:
                for example in self.examples:
                    messages += [{
                        'role': 'user',
                        'content': example[0] 
                    }, {
                        'role': 'assistant',
                        'content': example[1]
                    }]

            messages += [{
                'role': 'user',
                'content': self.wrap_task(task) + self.question_1st_stage
            }, {
                'role': 'assistant',
                'content': self.label_to_text_1st_stage[1]
            }, {
                'role': 'user',
                'content': self.question_2nd_stage
            }]
            return messages
        else:
            raise NotImplementedError(f'Invalid mode {mode}')
        
    def build_retry_two_stage_2(self, task: Task, prev_response: str, mode: Literal['openai', 'textgen'] = 'textgen') -> dict[str, any] | list[dict[str, str]]:
        if mode == 'textgen':
            if self.confirmation is not None:
                history = [self.confirmation]
            else:
                history = []
            if self.examples:
                history += self.examples
            history += [[self.wrap_task(task) + self.question_1st_stage, self.label_to_text_1st_stage[1]]]
            history += [[self.question_2nd_stage, prev_response]]
            return dict(
                **DEFAULT_CHAT_PARAMS,
                user_input = self.retry_msg_2nd_stage,
                history = dict(
                    internal = history,
                    visible = history
                ),
                context_instruct = self.instruction,
            )
        elif mode == 'openai':
            messages = [{
                'role': 'system',
                'content': self.instruction,
            }]
            if self.confirmation is not None:
                messages += [{
                    'role': 'user',
                    'content': self.confirmation[0],
                }, {
                    'role': 'assistant',
                    'content': self.confirmation[1],
                }]

            if self.examples:
                for example in self.examples:
                    messages += [{
                        'role': 'user',
                        'content': example[0] 
                    }, {
                        'role': 'assistant',
                        'content': example[1]
                    }]

            messages += [{
                'role': 'user',
                'content': self.wrap_task(task) + self.question_1st_stage
            }, {
                'role': 'assistant',
                'content': self.label_to_text_1st_stage[1]
            }, {
                'role': 'user',
                'content': self.question_2nd_stage
            }, {
                'role': 'assistant',
                'content': prev_response
            }, {
                'role': 'user',
                'content': self.retry_msg_2nd_stage
            }]
            return messages
        else:
            raise NotImplementedError(f'Invalid mode {mode}')

    def execute(
        self,
        task: Task,
        api_endpoint: str = 'http://localhost:5000/api/v1/chat',
        mode: Literal['openai', 'textgen'] = 'textgen',
        model: str = 'gpt-4',
        verbose: bool = False,
    ):
        if mode == 'openai':
            assert model == 'gpt-4' or model == 'gpt-3.5-turbo'

        params = self.build_two_stage_1(task, mode)
        if mode == 'textgen':
            output = send_request(api_endpoint, params)
            if verbose:
                print('Comment:', task.text)
                print('A:', task.alternative_a)
                print('B:', task.alternative_b)
                print('True label:', self.label_to_text_2nd_stage[task.label])
                print('1st stage model output:', output)
            output = remove_delimiters(output)

            retry_output = output
            retry_cnt = 0
            while retry_output.lower() not in self.text_to_label_1st_stage:
                retry_cnt += 1
                params = self.build_retry_two_stage_1(task, output)
                # if retry_cnt > 1:
                #     params['regenerate'] = True
                retry_output = send_request(api_endpoint, params)
                if verbose:
                    print('1st stage retry output:', retry_output)
                retry_output = remove_delimiters(retry_output)
                if retry_cnt == 10:
                    return False, None, params
            output = retry_output
            
            if output.lower() == self.label_to_text_1st_stage[0].lower():
                return True, 0, params
            elif output.lower() == self.label_to_text_1st_stage[1].lower():
                params = self.build_two_stage_2(task)
                output = send_request(api_endpoint, params)
                if verbose:
                    print('2nd stage model output:', output)
                output = remove_delimiters(output)

                valid_outputs = {
                    text.lower(): label 
                    for label,text in self.label_to_text_2nd_stage.items() if label > 0
                }
                retry_output = output
                retry_cnt = 0
                while retry_output.lower() not in valid_outputs:
                    retry_cnt += 1
                    params = self.build_retry_two_stage_2(task, output)
                    # if retry_cnt > 1:
                    #     params['regenerate'] = True
                    retry_output = send_request(api_endpoint, params)
                    if verbose:
                        print('2nd stage retry output:', retry_output)
                    retry_output = remove_delimiters(retry_output)
                    if retry_cnt == 10:
                        return False, None, params
                output = retry_output

                return True, self.text_to_label_2nd_stage[output.lower()], params
            else:
                raise ValueError(f'Invalid 1st stage output {output}')
        elif mode == 'openai':
            output = openai.ChatCompletion.create(
                model = model, 
                max_tokens = 10,
                messages = params,
                temperature = 1.,
                top_p = .7,
            )['choices'][0]['message']['content']
            if verbose:
                print('Comment:', task.text)
                print('A:', task.alternative_a)
                print('B:', task.alternative_b)
                print('True label:', self.label_to_text_2nd_stage[task.label])
                print('1st stage model output:', output)
            output = remove_delimiters(output)

            retry_output = output
            retry_cnt = 0
            while retry_output.lower() not in self.text_to_label_1st_stage:
                retry_cnt += 1
                params = self.build_retry_two_stage_1(task, output)
                # if retry_cnt > 1:
                #     params['regenerate'] = True
                retry_output = openai.ChatCompletion.create(
                    model = model, 
                    max_tokens = 10,
                    messages = params,
                    temperature = 1.,
                    top_p = .7,
                )['choices'][0]['message']['content']
                if verbose:
                    print('1st stage retry output:', retry_output)
                retry_output = remove_delimiters(retry_output)
                if retry_cnt == 10:
                    return False, None, params
            output = retry_output
            
            if output.lower() == self.label_to_text_1st_stage[0].lower():
                return True, 0, params
            elif output.lower() == self.label_to_text_1st_stage[1].lower():
                params = self.build_two_stage_2(task)
                output = openai.ChatCompletion.create(
                    model = model, 
                    max_tokens = 10,
                    messages = params,
                    temperature = 1.,
                    top_p = .7,
                )['choices'][0]['message']['content']
                if verbose:
                    print('2nd stage model output:', output)
                output = remove_delimiters(output)
                
                valid_outputs = {
                    text.lower(): label 
                    for label,text in self.label_to_text_2nd_stage.items() if label > 0
                }
                retry_output = output
                retry_cnt = 0
                while retry_output.lower() not in valid_outputs:
                    retry_cnt += 1
                    params = self.build_retry_two_stage_2(task, output)
                    # if retry_cnt > 1:
                    #     params['regenerate'] = True
                    retry_output = openai.ChatCompletion.create(
                        model = 'gpt-4', 
                        max_tokens = 10,
                        messages = params,
                        temperature = 1.,
                        top_p = .7,
                    )['choices'][0]['message']['content']
                    if verbose:
                        print('2nd stage retry output:', retry_output)
                    retry_output = remove_delimiters(retry_output)
                    if retry_cnt == 10:
                        return False, None, params
                output = retry_output

                return True, self.text_to_label_2nd_stage[output.lower()], params
            else:
                raise ValueError(f'Invalid 1st stage output {output}')
        else:
            raise NotImplementedError(f'Invalid mode {mode}')

    def __str__(self) -> str:
        class_attributes = [attr for attr in dir(self) if not callable(getattr(self, attr))]
        return '\n'.join([f'{attr}:\n{getattr(self, attr)}' for attr in class_attributes])