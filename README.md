# LLM Preference

## Prompt template

Place the appropriate yaml file under the directory for each dataset.

**instruction**
This is the instruction message sent right after the system message.

**confirmation**
This is a confirmation dialog between the user and model, reaffirming its task.

**retry_msg**
This is the message that will be sent to the LLM if it responds incorrectly.

**comment**
This is how the comment will be wrapped in the prompt.

**label**
This is how each label is treated inside the LLM.


## Running the code
Example:

```bash
python run.py --dataset college_confidential --template inwon --use_example
```

This command will run the experiments with `college confidential` dataset, using `inwon` prompt and adding examples.