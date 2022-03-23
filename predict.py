import logging
import numpy as np
import wandb
import utils
import tqdm
import torch
import torch.nn.functional as F
from collections import defaultdict

from argparse import ArgumentParser
from torch import nn
from loader import TweetDataset
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, ByT5Tokenizer
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, ShuffleSplit
from models import *

import random
import math

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str, action='store')
    parser.add_argument('--arch', default='char_lstm',
                        choices=['char_pool', 'char_lstm', 'char_cnn', 'char_lstm_cnn',
                                 'bert', 'byt5'])
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', required=True, type=str, action='store')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--mdn', action='store_true', default=False)
    parser.add_argument('--reduce_layer', action='store_true', default=False)
    parser.add_argument('--num_confidence_bins', type=int, default=5)
    parser.add_argument('--entropy_confidence', action='store_true', default=False)

    args = parser.parse_args()
    args.batch_size = 1 if args.generate else args.batch_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = TweetDataset(data_dir=args.data_dir)
    # sample = torch.randperm(len(test_dataset))
    # test_dataset = Subset(test_dataset, sample[:5000])

    byte_tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-small')
    word_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizers = (byte_tokenizer, word_tokenizer)

    collate_fn = lambda instance: utils.pad_chars(instance, tokenizers, -1)
    test_iter = tqdm.tqdm(DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn))

    state = torch.load(args.model_path, map_location=device)

    criterion = state['criterion']
    model_arch = utils.get_arch(state['arch'])

    if args.mdn:
        args.num_gaussians = state['state_dict']['_head.pi_h.weight'].size(0)

    model = model_arch(args)
    model.load_state_dict(state['state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        distances = []
        distances_confidence = defaultdict(list)

        for batch in test_iter:
            eval_stats = utils.evaluate(batch, model, criterion, args, generate=args.generate, device=device)
            if args.mdn:
                test_loss, (test_distance, test_distance_confidence) = eval_stats
                for confidence_level, corresp_test_distance in test_distance_confidence.items():
                    distances_confidence[int(confidence_level)].extend(corresp_test_distance.tolist())

            else:
                test_loss, test_distance = eval_stats

            test_iter.set_description(f"test loss: {test_loss.item()}")
            distances.extend(test_distance.tolist())

        logger.info(f"test_mean: {np.mean(distances)}, test_median: {np.median(distances)}")

        if args.mdn:
            for confidence_level, corresp_test_distance_list in distances_confidence.items():
                test_mean_c = np.nan_to_num(np.mean(corresp_test_distance_list))
                test_median_c = np.nan_to_num(np.median(corresp_test_distance_list))
                logger.info(f"conf: " + str(confidence_level) + " - " + f"val_mean: {test_mean_c}")
                logger.info(f"conf: " + str(confidence_level) + " - " + f"val_median: {test_median_c}")


if __name__ == '__main__':
    main()
