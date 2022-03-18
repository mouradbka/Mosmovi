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
    parser.add_argument('--data_dir', required=True, type=str, action='store')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--mdn', action='store_true', default=False)
    parser.add_argument('--use_mixture', action='store_true', default=False)
    parser.add_argument('--num_confidence_bins', type=int, default=5)
    parser.add_argument('--entropy_confidence', action='store_true', default=False)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = TweetDataset(data_dir=args.data_dir)

    byte_tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-small')
    word_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizers = (byte_tokenizer, word_tokenizer)

    collate_fn = lambda instance: utils.pad_chars(instance, tokenizers, -1)
    test_iter = tqdm.tqdm(DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn))

    state = torch.load(args.model_path, map_location=device)

    criterion = state['criterion']

    model_arch = utils.get_arch(state['arch'])
    model = model_arch(args)
    model.load_state_dict(state['state_dict'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        distances = []
        distances_confidence = defaultdict(list)

        for batch in test_iter:
            #test_loss, test_distance = utils.evaluate(batch, model, criterion, device=device, generate=args.generate)
            eval_stats = utils.evaluate(batch, model, criterion, args.mdn,
                                        device=device,
                                        mdn_mixture=args.use_mixture,
                                        no_bins=args.num_confidence_bins,
                                        entropy_confidence=args.entropy_confidence,
                                        generate = args.generate)
            if args.mdn:
                test_loss, test_distance, test_distance_confidence = eval_stats
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
                wandb.log({"conf: " + str(confidence_level) + " - " + "val_mean": test_mean_c,
                           "conf: " + str(confidence_level) + " - " + "val_median": test_median_c})
                logger.info(f"conf: " + str(confidence_level) + " - " + f"val_mean: {test_mean_c}")
                logger.info(f"conf: " + str(confidence_level) + " - " + f"val_median: {test_median_c}")


if __name__ == '__main__':
    main()
