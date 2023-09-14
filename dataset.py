import pandas as pd

class Dataset:
    dataframe: pd.DataFrame
    
    def __init__(
        self,
        raw_data: pd.DataFrame,
        text_col: str,
        alternative_a_col: str,
        alternative_b_col: str,
        label_col: str,
    ):
        self.dataframe = raw_data[[text_col,alternative_a_col,alternative_a_col]]
        
        
        
    def pick_examples(self, min_text_length: int = 100):
        examples = []
        for label, subset in dataset.groupby('Label'):
            # find the shortest on to use for example
            valid_length = subset[subset['Raw_text'].str.len() > min_text_length]
            example = valid_length.iloc[valid_length['Raw_text'].str.len().argmin()]
            examples += [Task(
                text = example['Raw_text'], 
                alternative_a = example['Alt_a'],
                alternative_b = example['Alt_b'],
                label = label
            )]

    return examples