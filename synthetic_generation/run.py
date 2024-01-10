import rich.progress
from rich.rule import Rule
from rich import print
from pathlib import Path
import random
import yaml
import click
import json

import numpy as np
import pandas as pd
from prompt import Prompt

import openai

openai.api_key = "sk-111111111111111111111111111111111111111111111111"
openai.api_base = "http://0.0.0.0:5000/v1"
np.random.seed(0)

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

def choose_alternatives(df):
    random_row_index = np.random.choice(df.index)
    return (
        df.loc[random_row_index, "alternative_a"],
        df.loc[random_row_index, "alternative_b"],
    )


DATASET_TO_PATH = {
    "college_confidential": "../data/college_confidential/dataset.csv",
    "compsent": "../data/compsent/dataset.csv",
    "pixie": "../data/pixie/dataset.csv",
}

@click.command()
@click.argument("dataset_name", default="compsent")
@click.option("--num_rows", default=10000)
def run(
    dataset_name: str,
    num_rows: int,
):
    output_dir = Path(f"outputs/{dataset_name}")
    output_dir.mkdir(exist_ok=True, parents=True)
    template = f"templates/{dataset_name}.yaml"
    config = yaml.safe_load(open(template))
    df = pd.read_csv(DATASET_TO_PATH[dataset_name])

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
            filename = output_dir / f"{i:08d}.json"
            if not filename.is_file():
                print(Rule())
                alternative_a, alternative_b = choose_alternatives(df)
                label, output, prompt_kwargs, instruction, task = prompt.execute(
                    alternative_a=alternative_a, alternative_b=alternative_b
                )

                one_row = {
                    "Label": label,
                    "Alt_a": alternative_a,
                    "Alt_b": alternative_b,
                    "Raw_text": output,
                    "Instruction": instruction,
                    "Task": task,
                    **prompt_kwargs,
                }
                json.dump(one_row, open(filename,"w"), indent=2)
            progress.update(progress_task, advance=1)

if __name__ == "__main__":
    run()
