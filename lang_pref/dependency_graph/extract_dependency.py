import stanza 
import pandas as pd
from typing import List,Union
from utils.check_device import USE_GPU
import torch

stanza.download('en')
nlp = stanza.Pipeline(
                lang='en', 
                processors='tokenize,mwt,pos,lemma,depparse',
                verbose=0,
                tokenize_pretokenized=True,
                use_gpu=USE_GPU
        )

'''
Parses the dependency edge information of given text. Can handle multiple sentences
'''

def text_to_deps(text:str) -> List[List[Union[int,str]]]:
    doc = nlp(text)
    # res[0]
    offset = 0
    all_sents = []

    for one_sent in doc.to_dict():
        df = pd.DataFrame(one_sent)
        idxs = df[['id','head']]-1 + offset
        # need to mark the root of each sentence while maintainig unique ids
        idxs['head'] = idxs['head'].replace({offset-1:-1})
        idxs['text'] = df['text']
        idxs['deprel'] = df['deprel']
        all_sents.append(idxs)
        offset += len(one_sent)
    combined = pd.concat(all_sents)

    # remove the edge that points to root
    combined = combined[combined['head'] > -1]
    return pd.Series({
        'edges':torch.tensor(combined[['id','head']].values.T.tolist()),
        'edges_info':combined.deprel.values.tolist()
    })

