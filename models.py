import logging
import torch
import torch.nn.functional as F

from torch import nn

logger = logging.getLogger()


class CharModel(nn.Module):
    def __init__(self, args):
        super(CharModel, self).__init__()
        self._token_embed = nn.Embedding(256, 300, 255)
        self._ffn = nn.Linear(300, 2)

    def forward(self, chars):
        embed = self._token_embed(chars)
        pool = torch.mean(embed, dim=1)
        return self._ffn(pool)


class CharLSTMModel(nn.Module):
    def __init__(self, args):
        super(CharLSTMModel, self).__init__()
        self._token_embed = nn.Embedding(256, 300, 255)
        self._ffn = nn.Linear(600, 2)
        self._lstm = nn.LSTM(300,300,2,bidirectional=True,batch_first=True)

    def forward(self, chars):
        embed = self._token_embed(chars)
        context_embeds = self._lstm(embed)[0]
        pool = torch.mean(context_embeds, dim=1)
        return self._ffn(pool)


class CharCNNModel(nn.Module):
    def __init__(self, args):
        super(CharCNNModel, self).__init__()
        self._token_embed = nn.Embedding(256, 300, 255)

        self._conv1 = nn.Sequential(
            nn.Conv1d(300, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self._conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self._conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self._conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self._conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self._conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
        )
        self._fc1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p=args.dropout))
        self._fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=args.dropout))
        self._fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self._token_embed(x)
        # transpose
        x = x.permute(0, 2, 1)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)
        x = self._conv6(x).squeeze()
        # linear layer
        x = self._fc1(x)
        # linear layer
        x = self._fc2(x)
        # final linear layer
        x = self._fc3(x)
        return x


class CharLSTMCNNModel(nn.Module):
    def __init__(self, args):
        super(CharCNNModel, self).__init__()
        self._token_embed = nn.Embedding(256, 300, 255)
        self._lstm = nn.LSTM(300,300,2,bidirectional=True,batch_first=True)

        self._conv1 = nn.Sequential(
            nn.Conv1d(600, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self._conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self._conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self._conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self._conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self._conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
        )
        self._fc1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p=args.dropout))
        self._fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=args.dropout))
        self._fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self._token_embed(x)
        x = self._lstm(x)[0]
        # transpose
        x = x.permute(0, 2, 1)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)
        x = self._conv6(x).squeeze()
        # linear layer
        x = self._fc1(x)
        # linear layer
        x = self._fc2(x)
        # final linear layer
        x = self._fc3(x)
        return x