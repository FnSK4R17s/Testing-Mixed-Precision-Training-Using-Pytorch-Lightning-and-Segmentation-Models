import torch.nn as nn
from torch.nn import functional as F

class Upscale34(nn.Module):
    def __init__(self):
        super(Upscale34, self).__init__()

        self.l0 = nn.Linear(1024, 2048)

    def forward(self, x):
        bs, _ = x.shape
        l0 = self.l0(x)

        return l0