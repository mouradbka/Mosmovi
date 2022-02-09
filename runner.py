import logging
import numpy as np
import wandb
import utils
import tqdm
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch import nn
from loader import TweetDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, ShuffleSplit
from models import *

import random
import math

logger = logging.getLogger()


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='./data', action='store')
    parser.add_argument('--model_type', default='char_lstm', choices=['char_pool', 'char_lstm',
                                                                      'char_cnn', 'char_lstm_cnn',
                                                                      'char_transformer'])
    parser.add_argument('--loss', default='mse', choices=['mse', 'mae', 'smooth_l1'])
    parser.add_argument('--split_uids', action='store_true')
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'SGD'])
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--subsample_ratio', default=None)
    parser.add_argument('--run_name', default=None)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project='mosmovi_1', config=args, name=args.run_name)

    tweet_dataset = TweetDataset(data_dir=args.data_dir)

    if args.split_uids:
        gss = GroupShuffleSplit(n_splits=1, train_size=0.9)
        train_indices, val_indices = next(gss.split(np.arange(len(tweet_dataset)), groups=tweet_dataset.uids))
    else:
        ss = ShuffleSplit(n_splits=1, train_size=0.9)
        train_indices, val_indices = next(ss.split(np.arange(len(tweet_dataset))))

    train_dataset = Subset(tweet_dataset, train_indices)
    val_dataset = Subset(tweet_dataset, val_indices)

    if args.subsample_ratio:
        train_dataset, val_dataset = utils.subsample_datasets(train_dataset, val_dataset, ratio=args.subsample_ratio)

    _train_iter = DataLoader(train_dataset, batch_size=int(args.batch_size), collate_fn=lambda x: utils.pad_chars(x))
    _val_iter = DataLoader(val_dataset, batch_size=int(args.batch_size), collate_fn=lambda x: utils.pad_chars(x))
    train_iter = tqdm.tqdm(_train_iter)
    val_iter = tqdm.tqdm(_val_iter)

    if args.model_type == 'char_pool':
        model = CharModel(args)
    elif args.model_type == 'char_lstm':
        model = CharLSTMModel(args)
    elif args.model_type == 'char_cnn':
        model = CharCNNModel(args)
    elif args.model_type == 'char_lstm_cnn':
        model = CharLSTMCNNModel(args)
    elif args.model_type == 'char_transformer':
        model = TransformerModel(args)

    model.to(device)
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'mae':
        criterion = nn.L1Loss()
    elif args.loss == 'smooth_l1':
        criterion = nn.SmoothL1Loss()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr))
    model.train()

    for epoch in range(10):
        for batch in train_iter:
            train_loss = utils.train(batch, model, optimizer, criterion, device=device)
            train_iter.set_description(f"train loss: {train_loss.item()}")
            wandb.log({"train_loss": train_loss.item()})

        with torch.no_grad():
            distances = []
            for batch in val_iter:
                val_loss, val_distance = utils.evaluate(batch, model, criterion, device=device)
                val_iter.set_description(f"validation loss: {val_loss.item()}")
                wandb.log({"val_loss": val_loss.item(), "val_distance": torch.mean(val_distance)})
                distances.extend(val_distance.tolist())
            wandb.log({"val_mean": np.mean(distances), "val_median": np.median(distances)})


if __name__ == '__main__':
    main()
