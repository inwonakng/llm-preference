from __future__ import annotations
import yaml

from chat_api import DEFAULT_CHAT_PARAMS, send_request

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
        self.text_to_label = {v:k for k,v in label_to_text.items()}

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

    def build(self, task: Task) -> dict[str, any]:
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
    
    def build_retry(self, task: Task, prev_response: str) -> dict[str, any]:
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

    def execute(
        self,
        task: Task,
        api_endpoint: str = 'http://localhost:5000/api/v1/chat',
    ) -> int:
        params = self.build(task)
        output = send_request(api_endpoint, params)
        print(task.text)
        print('A:', task.alternative_a)
        print('B:', task.alternative_b)
        print('True label:', self.label_to_text[task.label])
        print('Model output:', output)

        while output.lower() not in self.text_to_label:
            params = self.build_retry(task, output)
            output = send_request(api_endpoint, params)
            print('Retry output:', output)
        
        return self.text_to_label[output.lower()]

