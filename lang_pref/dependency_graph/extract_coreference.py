import torch
import pandas as pd
import numpy as np
from itertools import combinations
from allennlp.predictors.predictor import Predictor
from typing import List,Union
from utils.check_device import USE_GPU


predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
    cuda_device= 0 if USE_GPU else -1
)


