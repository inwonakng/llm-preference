import rich.progress
from rich.rule import Rule
from rich import print
from pathlib import Path
import random
import yaml

import numpy as np
import pandas as pd
from prompt import Prompt

import openai

CHECKPOINT_ROWS = 10


def progress_bar():
    return rich.progress.Progress(
        "[progress.description]{task.description}",
        rich.progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "{task.completed} of {task.total}",
        rich.progress.TimeRemainingColumn(),
        rich.progress.TimeElapsedColumn(),
        transient=True,
    )

def choose_alternatives_pixie(df):
    random_row_index = np.random.choice(df.index)
    return (
        df.loc[random_row_index, "alternative_a"],
        df.loc[random_row_index, "alternative_b"],
    )


DATASET_TO_PATH = {
    "college confidential": "../data/college_confidential/dataset.csv",
    "compsent": "../data/compsent/dataset.csv",
    "pixie": "../data/pixie/dataset.csv",
}
DATASET_TO_ALTERNATIVE_FUNCTION = {"pixie": choose_alternatives_pixie}


def run(template: str, num_rows: int, output_file: str):
    template = f"templates/{template}.yaml"
    config = yaml.safe_load(open(template))
    dataset = config["dataset"]
    df = pd.read_csv(DATASET_TO_PATH[dataset])
    alternatives_selector = DATASET_TO_ALTERNATIVE_FUNCTION[dataset]

    prompt = Prompt.load_template(
        template,
        examples=df.rename(columns={"text": "example"}),
    )

    rows = []

    with progress_bar() as progress:
        progress_task = progress.add_task(
            description="Generating rows...", total=num_rows
        )
        for i in range(num_rows):
            print(Rule())
            alternative_a, alternative_b = alternatives_selector(df)
            label, output, prompt_kwargs, instruction, task = prompt.execute(
                alternative_a=alternative_a, alternative_b=alternative_b
            )
            rows.append(
                {
                    "Label": label,
                    "Alt_a": alternative_a,
                    "Alt_b": alternative_b,
                    "Raw_text": output,
                    "Instruction": instruction,
                    "Task": task,
                    **prompt_kwargs,
                }
            )

            if (i + 1) % CHECKPOINT_ROWS == 0:
                pd.DataFrame(rows).to_csv(output_file, index=False)

            progress.update(progress_task, advance=1)

    pd.DataFrame(rows).to_csv(output_file, index=False)


if __name__ == "__main__":
    openai.api_key = "sk-111111111111111111111111111111111111111111111111"
    openai.api_base = "http://0.0.0.0:5000/v1"
    
    config = "pixie"
    run(config, 10000, f"{config}.csv")
