

import torch

def SAVE(model, PATH):
    torch.save(model, PATH)

def LOAD(PATH):
    model = torch.load(PATH)
    return model