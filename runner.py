import logging
import tqdm
import torch
import torch.nn.functional as F

from argparse import ArgumentParser
from torch import nn
from loader import TweetDataset
from torch.utils.data import DataLoader, random_split
from models import *

logger = logging.getLogger()


def pad_chars(instance):
    chars, coords = zip(*instance)
    pad_length = max(map(len, chars))
    padded_chars = [F.pad(i, (0, pad_length - len(i)), value=255) for i in chars]
    return torch.stack(padded_chars), torch.stack(coords)


def train(batch, model, optimizer, criterion):
    chars, coords = batch
    pred = model(chars)

    optimizer.zero_grad()
    loss = criterion(pred, coords)
    loss.backward()
    optimizer.step()

    return loss


def evaluate(batch, model, criterion):
    chars, coords = batch
    pred = model(chars)
    loss = criterion(pred, coords)

    return loss


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='./data', action='store')
    parser.add_argument('--model_type', default='char_pool', choices=['char_pool','char_lstm', 'char_cnn'])
    parser.add_argument('--loss', default='mse', choices=['mse','mae'])
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--data_ratio', default=0.1)
    args = parser.parse_args()

    tweet_dataset = TweetDataset(data_dir=args.data_dir)
    train_size = int(0.9 * len(tweet_dataset) * args.data_ratio)
    val_size = len(tweet_dataset) * args.data_ratio - train_size
    train_dataset, val_dataset = random_split(tweet_dataset, [train_size, val_size])

    _train_iter = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=lambda x: pad_chars(x))
    _val_iter = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=lambda x: pad_chars(x))
    train_iter = tqdm.tqdm(_train_iter)
    val_iter = tqdm.tqdm(_val_iter)

    if args.model_type == 'char_pool':
        model = CharModel(args)
    elif args.model_type == 'char_lstm':
        model = CharLSTMModel(args)
    elif args.model_type == 'char_cnn':
        model = CharCNNModel(args)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(10):
        for batch in train_iter:
            train_loss = train(batch, model, optimizer, criterion)
            train_iter.set_description(f"train loss: {train_loss.item()}")

        with torch.no_grad():
            val_losses = []
            for batch in val_iter:
                val_loss = evaluate(batch, model, criterion)
                val_iter.set_description(f"validation loss: {val_loss.item()}")
                val_losses.append(val_loss.item())

            # logger.warning(f'validation loss: {v}')


if __name__ == '__main__':
    main()
