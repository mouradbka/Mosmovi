import logging
import torch
import torch.nn.functional as F
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
        self._token_embed = nn.Embedding(256, 150, 255)
        self._ffn = nn.Linear(300, 2)
        self._lstm = nn.LSTM(150, 150, 2, bidirectional=True, batch_first=True)

    def forward(self, byte_tokens, word_tokens):
        input_ids = byte_tokens.input_ids
        embed = self._token_embed(input_ids)
        context_embeds = self._lstm(embed)[0]
        pool = torch.mean(context_embeds, dim=1)
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
        self._fc3 = nn.Linear(64, 2)

    def forward(self, byte_tokens, word_tokens):
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
            nn.ReLU()
        )
        self._conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
        )
        self._conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        if hasattr(args, 'dropout'):
            self.dropout = args.dropout
        else:
            self.dropout = 0.0

        self._fc1 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p=self.dropout))
        self._fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=self.dropout))
        self._fc3 = nn.Linear(128, 2)

    def forward(self, byte_tokens, word_tokens):
        input_ids = byte_tokens.input_ids
        x = self._token_embed(input_ids)
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


class TransformerLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward """
    def __init__(self, size, self_attn, feed_forward, dropout,
                 intermediate_layer_predictions=False, max_sequence_len=1024):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.add_positional_encoding = AddPositionalEncoding(size, max_sequence_len)
        self.norm = self.sublayer[0].norm
        self.size = size

    def forward(self, x, mask):
        x = self.add_positional_encoding(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x, None


class TransformerEncoder(nn.Module):
    """Core encoder is a stack of N layers"""
    def __init__(self, layer, n_layers, intermediate_layer_predictions=False):
        super(TransformerEncoder, self).__init__()
        self.layers = clones(layer, n_layers)
        # enforce a prediction for the last layer
        self.layers[-1].force_prediction = True
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x, prediction = layer(x, mask)
        return x #self.norm(x)#, intermediate_predictions


class TransformerModel(nn.Module):
    def __init__(self, args, n_layers=4,
                 hidden_size=128, inner_linear=256,
                 n_heads=4, dropout=0.55,
                 intermediate_layer_predictions=False):
        super(TransformerModel, self).__init__()

        attn = MultiHeadedAttention(n_heads, hidden_size, dropout)
        ff = PositionwiseFeedForward(hidden_size, inner_linear, dropout)

        self.encoder = TransformerEncoder(TransformerLayer(hidden_size, copy.deepcopy(attn), copy.deepcopy(ff),
                                            dropout, intermediate_layer_predictions, #generator,
                                            int(args.max_seq_len)),
                               n_layers, intermediate_layer_predictions)

        self._token_embed = nn.Embedding(256, 128, 255)
        self._ffn = nn.Linear(128, 2)

        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
             if p.dim() > 1:
                 nn.init.xavier_uniform_(p)

        self.intermediate_layer_predictions = intermediate_layer_predictions
        self.n_layers = n_layers

    def forward(self, byte_tokens, word_tokens):
        input_ids = byte_tokens.input_ids
        attention_mask = byte_tokens.attention_mask
        """Take in and process masked src and target sequences."""
        embeds = self._token_embed(input_ids)
        x = self.encoder(embeds, attention_mask)
        x = torch.mean(x, dim=1)

        return self._ffn(x)


class BertRegressor(nn.Module):
    def __init__(self, args):
        super(BertRegressor, self).__init__()
        self._model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self._head = nn.Linear(self._model.config.hidden_size, 2)
        # freeze whole model
        if args.freeze_layers == -1:
            for param in self._model.parameters():
                param.requires_grad = False
        # freeze part of model
        else:
            for l in range(args.freeze_layers):
                for name, param in self._model.named_parameters():
                    if name.startswith("mbert.encoder.layer." + str(l)):
                        param.requires_grad = False

    def forward(self, byte_tokens, word_tokens):
        out = self._model(**word_tokens)
        return self._head(out.pooler_output)


class ByT5Regressor(nn.Module):
    T5_HIDDEN_SIZE = 1472

    def __init__(self, args):
        super(ByT5Regressor, self).__init__()
        self._model = T5EncoderModel.from_pretrained('google/byt5-small')
        self._head = nn.Linear(self.T5_HIDDEN_SIZE, 2)
        #freeze whole model
        if args.freeze_layers == -1:
            for param in self._model.parameters():
                    param.requires_grad = False
        #freeze part of model
        else:
            for l in range(args.freeze_layers):
                for name, param in self._model.named_parameters():
                    if name.startswith("byt5.encoder.layer." + str(l)):
                        param.requires_grad = False

    def forward(self, byte_tokens, word_tokens):
        output = self._model(**byte_tokens)
        pooled_output = F.adaptive_max_pool1d(output.last_hidden_state.permute(0, 2, 1), output_size=1).squeeze()
        return self._head(pooled_output)
