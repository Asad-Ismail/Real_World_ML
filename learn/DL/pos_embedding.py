import torch.nn as nn
import math


class sinusoidPE():
    def __init__(self,max_len=1024,model_dim=512) -> None:
        super().__init__()
        pe= torch.zeros(max_len,model_dim)
        div_factor = torch.exp(torch.arange(0,model_dim,2)*(-math.log(10000.0)/model_dim)
        pos = torch.arange(max_len)

        sin= torch.sin(pos*dix)