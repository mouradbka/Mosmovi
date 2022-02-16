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
        self.coords = []
        self.uids = []
        self.char_max_length = char_max_length

        for fname in glob.glob(f"{data_dir}/*"):
            df = pd.read_csv(fname, sep=';', header=0)
            self.tweet_tokens.extend([str(i) for i in df.text.tolist()])
            self.uids.extend(df.author_id.tolist())
            self.coords.extend([tuple(map(float, fname.rstrip('.csv').split('/')[-1].split('_')))] * len(df))

    def __len__(self: Dataset) -> int:
        assert len(self.tweet_tokens) == len(self.coords)
        return len(self.tweet_tokens)

    def __getitem__(self: Dataset, idx: int):
        tokens = self.tweet_tokens[idx]
        coords = torch.FloatTensor(self.coords[idx])
        return tokens, coords
