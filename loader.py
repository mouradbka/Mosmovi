import torch
import logging
import pandas as pd
import glob
from torch.utils.data import Dataset
import numpy as np
logger = logging.getLogger()


class TweetDataset(Dataset):
    def __init__(self: Dataset, data_dir: str, char_max_length=1014) -> None:
        self.tweet_tokens = []
        self.tweet_chars = []
        self.coords = []
        self.uids = []
        self.char_max_length = char_max_length

        for fname in glob.glob(f"{data_dir}/*"):
            df = pd.read_csv(fname, sep=';', header=0)
            self.tweet_tokens.extend([str(i).split(" ") for i in df.text.tolist()])
            self.tweet_chars.extend([list(bytes(str(i), encoding='utf8')) for i in df.text.tolist()])
            self.uids.extend(df.author_id.tolist())
            self.coords.extend([tuple(map(float, fname.rstrip('.csv').split('/')[-1].split('_')))] * len(df))

    def __len__(self: Dataset) -> int:
        assert len(self.tweet_tokens) == len(self.tweet_chars) == len(self.coords)
        return len(self.tweet_chars)

    def __getitem__(self: Dataset, idx: int):
        chars = torch.LongTensor(self.tweet_chars[idx])
        coords = torch.FloatTensor(self.coords[idx])
        return chars, coords
