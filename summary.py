from __future__ import annotations
import yaml
import tiktoken
from chat_api import DEFAULT_CHAT_PARAMS, send_request
import json

import pandas as pd
encoding = tiktoken.get_encoding("cl100k_base")
DEFAULT_CHAT_PARAMS['max_new_tokens'] = 400
DEFAULT_CHAT_PARAMS['temperature'] = 0.3
DEFAULT_CHAT_PARAMS['top_p'] = 1

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

class Prompt_summary:
    instruction: str 
    retry_msg: str
    task_template: str
    confirmation: list[str] | None = None
    examples: list[list[str]] = []
    synonyms: dict[str, list[str]] 
    label_to_text: dict[int, str]
    def __init__(
        self,
        instruction: str, 
        retry_msg: str,
        task_template: str,
        examples: list, 
        synonyms: dict[str, list[str]],
        label_to_text: dict[int, str],
    ) -> None:
        self.instruction = instruction 
        self.retry_msg =  retry_msg
        self.task_template = task_template
        self.examples = [] 
        self.synonyms = {} 
        self.label_to_text = label_to_text
        self.text_to_label = {
            text.lower(): label 
            for label,text in label_to_text.items()
        }
    @staticmethod
    def load_template(template_path: str) -> Prompt_summary:
        config = yaml.safe_load(open(template_path))
        prompt = Prompt_summary(
            instruction = config['instruction_summary'], 
            retry_msg = config['retry_msg_summary'],
            task_template = config['task'],
            label_to_text = config['label'],
            examples = [],
            synonyms = {},
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
    
    '''load example for summarization''' 
    def add_examples_summary(self):
        examples = json.load(open('./summary/example_summary.json'))
        examples_task = []
        for e in examples['examples']:
            e = Task(e['text'], e['label'], e['alternative_a'], e['alternative_b'])
            examples_task.append(e)

        self.examples = [
            [
                self.wrap_task(e),
                e.label
            ]
            for e in examples_task
        ]

    '''load university synonyms'''    
    def load_synonyms(self):
        synonyms = yaml.safe_load(open("./summary/synonyms.json"))
        for i in synonyms.keys():
            synonyms[i] += [i]
        self.synonyms = synonyms

    def build(self, task: Task) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        if self.examples:
            history += self.examples
        params = dict(
            **DEFAULT_CHAT_PARAMS,
            user_input = self.wrap_task(task),
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )

        return params
    
    def build_retry(self, task: Task, prev_response: str) -> dict[str, any]:
        if self.confirmation is not None:
            history = [self.confirmation]
        else:
            history = []
        if self.examples:
            history += self.examples            
        history += [[self.wrap_task(task), prev_response]]

        params = dict(
            **DEFAULT_CHAT_PARAMS,
            user_input = self.retry_msg,
            history = dict(
                internal = history,
                visible = history
            ),
            context_instruct = self.instruction,
        )

        return params
    
    def check_choices_exist(
        self, 
        alternative_a: list[str],
        alternative_b: list[str],
        output: str
    ) -> bool:

        check_a = False
        check_b = False
        for i in alternative_a:
            if i in output:
                check_a = True

        for i in alternative_b:
            if i in output:
                check_b = True

        return not check_a or not check_b # both choices exist in summary
        # return not check_a and not check_b # one choice exists in summary

    def update_examples(
        self, 
        ori_text: str,
        alternative_a: str,
        alternative_b: str,
        output_summary: str,
        retry_cnt: int,
    ):  
        if retry_cnt < 5:
            e = Task(ori_text, output_summary, alternative_a, alternative_b)
            self.examples += [[
                self.wrap_task(e),
                e.label
            ]]

        if len(self.examples) > 10:
            self.examples.pop(0)

    def checK_len_summary(
        self, 
        output_summary_temp: str,
        text: str,
    ):
        summary_too_long = False
        len_summary = len(encoding.encode(output_summary_temp))
        len_text = len(encoding.encode(text))
        if len_text < 50:
            if len_summary > 40: 
                summary_too_long = True
        elif 50 <= len_text < 100:
            if len_summary > 80: 
                summary_too_long = True
        elif 100 <= len_text :
            if len_summary > 150:
                summary_too_long = True
        return summary_too_long

    def execute(
        self,
        task: Task,
        api_endpoint: str = 'http://localhost:5000/api/v1/chat',
        verbose: bool = False,
    ) -> int:
        params = self.build(task)
        output = send_request(api_endpoint, params)
        output = remove_delimiters(output)
        alternative_a = self.synonyms[task.alternative_a]
        alternative_b = self.synonyms[task.alternative_b]
        choices_in_summary = self.check_choices_exist(alternative_a, alternative_b, output)

        # retry
        retry_cnt = 0
        output_temp = output
        summary_too_long = self.checK_len_summary(output_temp, task.text)

        while ( choices_in_summary or summary_too_long ) and retry_cnt < 5:
            retry_cnt += 1
            params = self.build_retry(task, output_temp)
            output_temp = send_request(api_endpoint, params)
            output_temp = remove_delimiters(output_temp)
            choices_in_summary = self.check_choices_exist(alternative_a, alternative_b, output_temp)
        if retry_cnt < 5: 
            output = output_temp # retry success
        print(f'Summary :{output}')

        return output, retry_cnt

    def __str__(self) -> str:
        class_attributes = [attr for attr in dir(self) if not callable(getattr(self, attr))]
        return '\n'.join([f'{attr}:\n{getattr(self, attr)}' for attr in class_attributes])