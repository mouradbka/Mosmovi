import math
import sys
import shutil
import random
import torch
import torch.nn.functional as F

from torch import nn, optim
from models import *


EARTH_RADIUS = 6372.8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_arch(arch):
    archs = {
        'char_pool': CharModel,
        'char_lstm': CharLSTMModel,
        'char_cnn': CharCNNModel,
        'char_lstm_cnn': CharLSTMCNNModel,
        'char_transformer': TransformerModel,
        'bert': BertRegressor,
        'byt5': ByT5Regressor,
    }
    return archs[arch]


def get_criterion(crit):
    crits = {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss(),
    }
    return crits[crit]


def get_optimizer(opt):
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
    }
    return optimizers[opt]


def gc_distance(gold, pred):
    _degree_radian = lambda d: (d * math.pi) / 180
    rad_gold = _degree_radian(gold)
    rad_pred = _degree_radian(pred)

    cos_gold = torch.cos(rad_gold)
    sin_gold = torch.sin(rad_gold)
    cos_pred = torch.cos(rad_pred)
    sin_pred = torch.sin(rad_pred)

    n_gold = torch.stack([cos_gold[:, 0] * cos_gold[:, 1], cos_gold[:, 0] * sin_gold[:, 1], sin_gold[:, 0]], dim=1)
    n_pred = torch.stack([cos_pred[:, 0] * cos_pred[:, 1], cos_pred[:, 0] * sin_pred[:, 1], sin_pred[:, 0]], dim=1)

    return torch.nan_to_num(torch.acos(torch.inner(n_gold.to(device), n_pred.to(device)).diag()) * EARTH_RADIUS)


def pad_chars(instance, tokenizers, max_length=-1):
    tokens, coords = zip(*instance)
    byte_tokenizer, word_tokenizer = tokenizers

    word_tokens = word_tokenizer(tokens, padding=True, return_tensors='pt')

    if max_length == -1:
        byte_tokens = byte_tokenizer(tokens, padding=True, return_tensors='pt')
        if byte_tokens.input_ids.size(1) < 7:
            byte_tokens = byte_tokenizer(tokens, padding='max_length', max_length=7, return_tensors='pt')
    else:
        byte_tokens = byte_tokenizer(tokens, truncation=True, padding='max_length', max_length=max_length,
                                     return_tensors='pt')

    return byte_tokens, word_tokens, torch.stack(coords)


def subsample_datasets(train_dataset, val_dataset, ratio):
    train_list = random.sample(range(len(train_dataset)), int(math.ceil(len(train_dataset) * float(ratio))))
    val_list = random.sample(range(len(val_dataset)), int(math.ceil(len(val_dataset) * float(ratio))))
    train_dataset = torch.utils.data.Subset(train_dataset, train_list)
    val_dataset = torch.utils.data.Subset(val_dataset, val_list)
    return train_dataset, val_dataset


def train(batch, model, optimizer, scheduler, criterion, device):
    byte_tokens, word_tokens, coords = batch

    pred = model(byte_tokens.to(device), word_tokens.to(device))
    optimizer.zero_grad()
    loss = criterion(pred, coords.to(device))
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss


def evaluate(batch, model, criterion, device, generate=False):
    byte_tokens, word_tokens, coords = batch
    # check if batch dim squeezed out during pred, fix
    pred = model(byte_tokens.to(device), word_tokens.to(device))

    if len(pred.shape) == 1:
        pred = pred[None, :]
    loss = criterion(pred, coords.to(device))
    distance = gc_distance(coords, pred)

    if generate:
        assert len(byte_tokens.input_ids) == len(pred) == 1
        # use lengths to avoid having strings shorter than 7 bytes
        tweet = bytes(byte_tokens.input_ids[0, :-1] - 3).decode('utf-8')
        lat, long = pred[0][0], pred[0][1]
        sys.stdout.write(f"{tweet}\t({lat}, {long})\n")

    return loss, distance


def split_users(df, percentage_users=0.10):
    users_ids = df.author_id
    users_ids = set(df.author_id)
    n_samples = int(len(users_ids)*percentage_users)
    val_users = random.sample(users_ids, n_samples)
    return df.loc[df.author_id.isin(val_users)]
