import logging
import wandb
import utils
import tqdm
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch import nn
from loader import TweetDataset
from torch.utils.data import DataLoader, random_split
from models import *

import random
import math

logger = logging.getLogger()


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='./data', action='store')
    parser.add_argument('--model_type', default='char_pool', choices=['char_pool', 'char_lstm', 'char_cnn'])
    parser.add_argument('--loss', default='mse', choices=['mse', 'mae'])
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--subsample_ratio', default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project='mosmovi_1', config=args)

    tweet_dataset = TweetDataset(data_dir=args.data_dir)
    if args.subsample_ratio:
        subsample_list = random.sample(range(len(tweet_dataset)), int(math.ceil(len(tweet_dataset) * float(args.subsample_ratio))))
        tweet_dataset = torch.utils.data.Subset(tweet_dataset, subsample_list)

    train_size = int(0.9 * len(tweet_dataset))
    val_size = len(tweet_dataset) - train_size
    train_dataset, val_dataset = random_split(tweet_dataset, [train_size, val_size])

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

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()

    for epoch in range(10):
        for batch in train_iter:
            # pass
            train_loss = utils.train(batch, model, optimizer, criterion, device=device)
            train_iter.set_description(f"train loss: {train_loss.item()}")
            wandb.log({"train_loss": train_loss.item()})

        with torch.no_grad():
            for batch in val_iter:
                val_loss, val_distance = utils.evaluate(batch, model, criterion, device=device)
                val_iter.set_description(f"validation loss: {val_loss.item()}")
                wandb.log({"val_loss": val_loss.item(), "val_distance": torch.mean(val_distance)})


if __name__ == '__main__':
    main()
