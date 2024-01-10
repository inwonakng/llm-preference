from __future__ import annotations

import yaml
from typing import List, Dict
from chat_api import DEFAULT_CHAT_PARAMS
import random
import pandas as pd
import openai


class Prompt:
    instruction: str
    retry_msg: str
    task_template: str
    examples: pd.DataFrame = None

    def __init__(
        self,
        instruction_template: str,
        task_template: str,
        labels: Dict[int, str],
        example_template: str = None,
        num_examples: int = 0,
        examples: pd.DataFrame = None,
        options: Dict = None,
    ) -> None:
        self.instruction_template = instruction_template
        self.task_template = task_template
        self.labels = labels
        self.example_template = example_template
        self.num_examples = num_examples
        self.examples = examples
        self.options = options

    @staticmethod
    def load_template(template_path: str, examples: pd.DataFrame = None) -> Prompt:
        """Method to create a Prompt from a template and examples dataframe

        template_path (str): path to the template that the prompt should follow
        examples (pd.DataFrame): dataframe that contains the examples, "label" and "example" should be included in the columns
        """
        config = yaml.safe_load(open(template_path))
        prompt = Prompt(
            instruction_template=config["instruction"],
            task_template=config["task"],
            labels=config["labels"],
            example_template=config.get("example", None),
            num_examples=config.get("num_examples", 0),
            examples=examples,
            options=config["options"],
        )
        return prompt

    def get_examples(self, label: int):
        if self.num_examples == 0:
            return ""

        examples_subset = self.examples[self.examples["label"] == label][
            "example"
        ].tolist()

        examples = random.sample(
            examples_subset,
            self.num_examples,
        )
        formatted_examples = []

        for idx, example in enumerate(examples):
            formatted_examples.append(
                self.example_template.format(example_idx=idx + 1, example=example)
            )

        return "\n".join(formatted_examples)

    def build(self, label: int, prompt_kwargs: Dict) -> Dict[str, any]:
        examples = self.get_examples(label)

        options_dict = (
            {key: random.choice(options) for key, options in self.options.items()}
            if self.options
            else {}
        )

        prompt_kwargs = {
            **prompt_kwargs,
            "examples": examples,
            **options_dict,
        }
        instruction = self.instruction_template.format(**prompt_kwargs)
        task = self.task_template.format(**prompt_kwargs)

        return (
            dict(
                **DEFAULT_CHAT_PARAMS,
                user_input=task,
                context_instruct=instruction,
            ),
            prompt_kwargs,
            instruction,
            task,
        )

    def label(self, messages):
        response = openai.ChatCompletion.create(model="x", messages=messages)

        return response["choices"][0]["message"]["content"]

    def execute(
        self,
        **prompt_kwargs,
    ) -> int:
        label, label_template = random.choice(list(self.labels.items()))
        prompt_kwargs["label_text"] = label_template.format(
            alternative_a=prompt_kwargs["alternative_a"],
            alternative_b=prompt_kwargs["alternative_b"],
        )
        params, prompt_kwargs, instruction, task = self.build(label, prompt_kwargs)
        output = self.label([{"role": "system", "content": instruction}, {"role": "user", "content": task}])
        print(params["context_instruct"])
        print(params["user_input"])
        print("Model output:", output)

        return label, output, prompt_kwargs, instruction, task
