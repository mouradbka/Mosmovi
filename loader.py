import torch
import json
import logging
import pandas as pd
import glob
from torch.utils.data import Dataset
logger = logging.getLogger()


class TweetDataset(Dataset):
    def __init__(self: Dataset, data_dir: str, char_max_length=1014, subsample=False) -> None:
        self.tweet_tokens = []
        self.dirnames = []
        self.coords = []
        self.labels = []
        self.uids = []
        self.char_max_length = char_max_length

        for fname in glob.glob(f"{data_dir}/*/*"):
            d = json.load(open(fname))
            tweets_in_frame = d['tw_meta']['result_count']
            dirname = fname.split('/')[-2]
            self.dirnames.extend([fname] * tweets_in_frame)
            self.tweet_tokens.extend([str(i['text']) for i in d['tw_data']])
            self.uids.extend(int(i['author_id']) for i in d['tw_data'])
            self.coords.extend([tuple(map(float, dirname.split('_')))] * tweets_in_frame)

    def __len__(self: Dataset) -> int:
        assert len(self.tweet_tokens) == len(self.coords)
        return len(self.tweet_tokens)

    def __getitem__(self: Dataset, idx: int):
        tokens = self.tweet_tokens[idx]
        labels = torch.FloatTensor(self.coords[idx])
        return tokens, labels
