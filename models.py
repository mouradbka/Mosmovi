import logging
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from transformers import BertModel, T5EncoderModel

from model_utils import *

logger = logging.getLogger()


class CharModel(nn.Module):
    def __init__(self, args):
        super(CharModel, self).__init__()
        self._token_embed = nn.Embedding(256, 300, 255)
        self._ffn = nn.Linear(300, 2)

    def forward(self, byte_tokens, word_tokens):
        input_ids = byte_tokens.input_ids
        embed = self._token_embed(input_ids)
        pool = torch.mean(embed, dim=1)
        return self._ffn(pool)


class CharLSTMModel(nn.Module):
    def __init__(self, args):
        super(CharLSTMModel, self).__init__()
        self._token_embed = nn.Embedding(256, 50, 255)
        self._ffn = nn.Linear(100, 2)
        self._lstm = nn.LSTM(50, 50, 2, bidirectional=True, batch_first=True)

    def forward(self, byte_tokens, word_tokens, features_only=False):
        input_ids = byte_tokens.input_ids
        embed = self._token_embed(input_ids)
        context_embeds = self._lstm(embed)[0]
        pool = torch.mean(context_embeds, dim=1)
        if features_only:
            return pool
        return self._ffn(pool)


class CharCNNModel(nn.Module):
    def __init__(self, args):
        super(CharCNNModel, self).__init__()
        self._token_embed = nn.Embedding(256, 150, 255)

        self._conv1 = nn.Sequential(
            nn.Conv1d(150, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        )
        self._conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        )
        self._conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self._conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self._conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self._conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        if hasattr(args, 'dropout'):
            self.dropout = args.dropout
        else:
            self.dropout = 0.0

        self._fc1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=self.dropout))
        self._fc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=self.dropout))
        if args.mdn:
            self._fc3 = MDN(64, 2, 10)
        else:
            self._fc3 = nn.Linear(64, 2)

    def forward(self, byte_tokens, word_tokens, features_only=False):
        input_ids = byte_tokens.input_ids
        x = self._token_embed(input_ids)
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
        if features_only:
            return x
        # final linear layer
        x = self._fc3(x)
        return x

class CharLSTMCNNModel(nn.Module):
    def __init__(self, args):
        super(CharLSTMCNNModel, self).__init__()
        self._token_embed = nn.Embedding(256, 150, 255)
        self._lstm = nn.LSTM(150,150,2,bidirectional=True,batch_first=True)

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
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self._conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        if hasattr(args, 'dropout'):
            self.dropout = args.dropout
        else:
            self.dropout = 0.0

        self._fc1 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=self.dropout))
        if args.mdn:
            self._fc2 = MDN(128, 2, 10)
        else:
            self._fc2 = nn.Linear(128, 2)

    def forward(self, byte_tokens, word_tokens):
        input_ids = byte_tokens.input_ids
        x = self._token_embed(input_ids)
        x = self._lstm(x)[0]
        # transpose
        x = x.permute(0, 2, 1)
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x).squeeze()
        # linear layer
        x = self._fc1(x)
        # final linear layer
        x = self._fc2(x)
        return x


class RBFLayer(nn.Module):
    def __init__(self, encoding_dim):
        super(RBFLayer, self).__init__()
        self.rbf = nn.parameter.Parameter(torch.FloatTensor(range(encoding_dim)) / encoding_dim)
        self.sigma = nn.parameter.Parameter(torch.ones(encoding_dim) * np.sqrt(0.5 / encoding_dim))

    def forward(self, time):
        return torch.exp((-(time - self.mu) ** 2) / (2 * (self.sigma ** 2)))


class BertRegressor(nn.Module):
    def __init__(self, args):
        super(BertRegressor, self).__init__()
        self._model = BertModel.from_pretrained('bert-base-multilingual-cased')
        # freeze whole model
        if args.freeze_layers == -1:
            for param in self._model.parameters():
                param.requires_grad = False

        # freeze part of model
        else:
            for l in range(args.freeze_layers):
                for name, param in self._model.named_parameters():
                    if name.startswith("encoder.layer." + str(l)):
                        param.requires_grad = False

    def forward(self, byte_tokens, word_tokens):
        return self._model(**word_tokens).pooler_output


class ByT5Regressor(nn.Module):
    def __init__(self, args):
        super(ByT5Regressor, self).__init__()
        self._model = T5EncoderModel.from_pretrained('google/byt5-small')
        # freeze whole model
        if args.freeze_layers == -1:
            for param in self._model.parameters():
                    param.requires_grad = False
        # freeze part of model
        else:
            for l in range(args.freeze_layers):
                for name, param in self._model.named_parameters():
                    if name.startswith("encoder.block." + str(l)):
                        param.requires_grad = False

    def forward(self, byte_tokens, word_tokens):
        output = self._model(**byte_tokens)
        return F.adaptive_max_pool1d(output.last_hidden_state.permute(0, 2, 1), output_size=1).squeeze()


class CompositeModel(nn.Module):
    T5_HIDDEN_SIZE = 1472

    def __init__(self, args):
        super(CompositeModel, self).__init__()

        self.use_metadata = args.use_metadata
        if args.arch == 'bert':
            self._encoder = BertRegressor(args)
            concat_dim = self._encoder._model.config.hidden_size
        elif args.arch == 'byt5':
            self._encoder = ByT5Regressor(args)
            concat_dim = self.T5_HIDDEN_SIZE

        if args.use_metadata:
            self._tweet_rbf = nn.Linear(1, args.tweet_rbf_dim) #RBFLayer(encoding_dim=args.tweet_rbf_dim)
            self._author_rbf = nn.Linear(1, args.author_rbf_dim) #RBFLayer(encoding_dim=args.author_rbf_dim)
            self._description_lstm = CharLSTMModel(args)
            concat_dim += args.tweet_rbf_dim + args.author_rbf_dim + (self._description_lstm._lstm.hidden_size * 2)

        self._reduce = nn.Linear(concat_dim, 100)
        if args.mdn:
            self._head = MDN(100, 2, 20)
        else:
            self._head = nn.Linear(100, 2)

    def forward(self, byte_tokens, word_tokens, metadata):
        text_encoding = self._encoder(byte_tokens, word_tokens)
        if self.use_metadata:
            tweet_time, author_time, author_desc = metadata
            encoded_tweet_time = self._tweet_rbf(tweet_time)
            encoded_author_time = self._author_rbf(author_time)
            encoded_desc = self._description_lstm(author_desc, None, features_only=True)
            concat = torch.cat([text_encoding, encoded_desc, encoded_tweet_time, encoded_author_time], dim=-1)
        else:
            concat = text_encoding

        return self._head((self._reduce(F.dropout(concat, p=0.2))))


