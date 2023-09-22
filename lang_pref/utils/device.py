import torch

USE_GPU = torch.cuda.is_available()
if USE_GPU: print('We are using GPU! 🏎️ 🏎️')
