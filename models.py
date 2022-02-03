import logging
import torch
import torch.nn.functional as F

from torch import nn

logger = logging.getLogger()


class CharModel(nn.Module):
    def __init__(self):
        super(CharModel, self).__init__()
        self._token_embed = nn.Embedding(256, 300, 255)
        self._ffn = nn.Linear(300, 2)

    def forward(self, chars):
        embed = self._token_embed(chars)
        pool = torch.mean(embed, dim=1)
        return self._ffn(pool)


