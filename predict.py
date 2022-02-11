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
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str, action='store')
    parser.add_argument('--data_dir', required=True, type=str, action='store')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--batch_size', default=128)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = TweetDataset(data_dir=args.data_dir)
    if args.generate:
        test_iter = tqdm.tqdm(DataLoader(test_dataset, batch_size=1, collate_fn=lambda x: utils.pad_chars(x)))
    else:
        test_iter = tqdm.tqdm(DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=lambda x: utils.pad_chars(x)))

    state = torch.load(args.model_path, map_location=device)

    criterion = state['criterion']

    model_arch = utils.get_arch(state['arch'])
    model = model_arch(args)
    model.load_state_dict(state['state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        distances = []
        for batch in test_iter:
            test_loss, test_distance = utils.evaluate(batch, model, criterion, device=device, generate=args.generate)
            test_iter.set_description(f"test loss: {test_loss.item()}")
            distances.extend(test_distance.tolist())
        logger.info(f"test_mean: {np.mean(distances)}, test_median: {np.median(distances)}")


if __name__ == '__main__':
    main()
