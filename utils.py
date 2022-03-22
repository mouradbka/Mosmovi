import math
import sys
import shutil
import random
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.stats import kurtosis

from torch import nn, optim
from models import *
from model_utils import mdn_loss, predict
from model_utils import sample as mdn_sample

import numpy as np

EARTH_RADIUS = 6372.8


def get_arch(arch):
    archs = {
        'char_pool': CharModel,
        'char_lstm': CompositeModel,
        'char_cnn': CharCNNModel,
        'char_lstm_cnn': CharLSTMCNNModel,
        'bert': CompositeModel,
        'byt5': CompositeModel,
    }
    return archs[arch]


def get_criterion(crit):
    crits = {
        'mse': nn.MSELoss(),
        'l1': nn.L1Loss(),
        'smooth_l1': nn.SmoothL1Loss(),
        'cross_entropy': nn.CrossEntropyLoss()
    }
    return crits[crit]


def get_optimizer(opt):
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
    }
    return optimizers[opt]


def split_users(df, percentage_users=0.10):
    users_ids = df.author_id
    users_ids = set(df.author_id)
    n_samples = int(len(users_ids)*percentage_users)
    val_users = random.sample(users_ids, n_samples)
    return df.loc[df.author_id.isin(val_users)]


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

    return torch.nan_to_num(torch.acos(torch.inner(n_gold.to(gold.device), n_pred.to(pred.device)).diag()) * EARTH_RADIUS)


def pad_chars(instance, tokenizers, max_length=-1):
    tokens, coords = zip(*instance)
    byte_tokenizer, word_tokenizer = tokenizers

    word_tokens = word_tokenizer(tokens, padding=True, return_tensors='pt', truncation=True)

    def tokenize_maybe_pad(tokenizer, tokens, length=7):
        tokenized = tokenizer(tokens, padding=True, return_tensors='pt')
        if tokenized.input_ids.size(1) < length:
            tokenized = tokenizer(tokens, padding='max_length', max_length=length, return_tensors='pt')
        return tokenized

    if max_length == -1:
        byte_tokens = tokenize_maybe_pad(byte_tokenizer, tokens)
    else:
        byte_tokens = byte_tokenizer(tokens, truncation=True, padding='max_length', max_length=max_length,
                                     return_tensors='pt')

    encoded_tokens = (byte_tokens, word_tokens)
    encoded_coords = torch.stack(coords)

    return encoded_tokens, encoded_coords


def subsample_datasets(train_dataset, val_dataset, ratio):
    train_list = random.sample(range(len(train_dataset)), int(math.ceil(len(train_dataset) * float(ratio))))
    val_list = random.sample(range(len(val_dataset)), int(math.ceil(len(val_dataset) * float(ratio))))
    train_dataset = torch.utils.data.Subset(train_dataset, train_list)
    val_dataset = torch.utils.data.Subset(val_dataset, val_list)
    return train_dataset, val_dataset


def train(batch, model, optimizer, scheduler, criterion, args, device):
    step, batch = batch
    encoded_tokens, coords = batch
    encoded_tokens = [i.to(device) for i in encoded_tokens]
    coords = coords.to(device)

    byte_tokens, word_tokens = encoded_tokens

    if args.mdn:
        pi, mu, sigma = model(byte_tokens, word_tokens)
        loss = mdn_loss(coords, pi, mu, sigma)
        # l1 penalty
        if args.reg_penalty > 0.0:
            sigma_params = torch.cat([x.view(-1) for x in model._head.sigma_h.parameters()])
            mu_params = torch.cat([x.view(-1) for x in model._head.mu_h.parameters()])

            sigma_penalty = args.reg_penalty * 0.5 * torch.sum(sigma_params ** 2)
            mu_penalty = args.reg_penalty * 0.5 * torch.sum(mu_params ** 2)

            loss += sigma_penalty + mu_penalty

        if args.entropy_loss_weight > 0.0:
            entropy_loss = -args.entropy_loss_weight * Categorical(pi).entropy().sum()
            loss += entropy_loss

    else:
        pred = model(byte_tokens, word_tokens)
        loss = criterion(pred, coords)

    loss.backward()
    if (step + 1) % args.gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return loss


def evaluate(batch, model, criterion, args, device, generate=False):
    encoded_tokens, coords = batch

    encoded_tokens = [i.to(device) for i in encoded_tokens]
    coords = coords.to(device)

    byte_tokens, word_tokens = encoded_tokens

    # check if batch dim squeezed out during pred, fix
    if args.mdn:
        pi, mu, sigma = model(byte_tokens, word_tokens)
        if args.use_mixture:
            pred = predict(pi, mu, sigma, method='mixture')
        else:
            pred = predict(pi, mu, sigma, method='pi')

        pred = pred.unsqueeze(0) if len(pred.size()) == 1 else pred

        # calculate confidence
        if args.entropy_confidence:
            entropy = Categorical(pi).entropy()
            max_val = np.max(entropy.cpu().detach().numpy())
            min_val = np.min(entropy.cpu().detach().numpy())
            bins = np.linspace(max_val, min_val, args.num_confidence_bins)
            confidence = np.digitize(entropy.cpu().detach().numpy(), bins)
        else:
            max_prob, max_idx = torch.max(pi, dim=1)
            max_val = np.max(max_prob.cpu().detach().numpy())
            min_val = np.min(max_prob.cpu().detach().numpy())
            bins = np.linspace(min_val, max_val, args.num_confidence_bins)
            confidence = np.digitize(max_prob.cpu().detach().numpy(), bins)

        distance = gc_distance(coords, pred)

        # calculate distance per confidence level
        confidence_distance = {}
        for confidence_level in list(set(confidence)):
            item_selector = [c == confidence_level for c in confidence]
            selected_coords, selected_preds = coords[item_selector, :], pred[item_selector, :]
            confidence_distance[confidence_level] = gc_distance(selected_coords, selected_preds)

        distance = distance, confidence_distance
    else:
        pred = model(byte_tokens, word_tokens)
        pred = pred.unsqueeze(0) if len(pred.size()) == 1 else pred
        distance = gc_distance(coords, pred)

    loss = criterion(pred, coords)
    if generate:
        assert len(byte_tokens.input_ids) == len(pred) == 1
        # use lengths to avoid having strings shorter than 7 bytes
        tweet = bytes(byte_tokens.input_ids[0, :-1] - 3).decode('utf-8')
        lat, long = pred[0][0], pred[0][1]
        outstring = f"{tweet}\t({lat}, {long})"
        if args.mdn:
            outstring += f"\t{confidence[0]}"

        sys.stdout.write(f"{outstring}\n")

    return loss, distance
