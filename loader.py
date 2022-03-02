import torch
import logging
import pandas as pd
import glob
from torch.utils.data import Dataset
import numpy as np
logger = logging.getLogger()
import hdbscan


class TweetDataset(Dataset):
    def __init__(self: Dataset, data_dir: str, char_max_length=1014, use_metadata=True, classify=False) -> None:
        self.tweet_tokens = []
        self.tweet_time = []
        self.author_time = []
        self.author_desc = []
        self.fnames = []
        self.coords = []
        self.uids = []
        self.char_max_length = char_max_length
        self.use_metadata = use_metadata
        self.classify = classify

        for fname in glob.glob(f"{data_dir}/*"):
            df = pd.read_csv(fname, sep=';', header=0)
            self.fnames.append(fname)
            self.tweet_tokens.extend([str(i) for i in df.text.tolist()])
            self.uids.extend(df.author_id.tolist())
            self.coords.extend([tuple(map(float, fname.rstrip('.csv').split('/')[-1].split('_')))] * len(df))
            if use_metadata:
                def get_normalised_time(dt):
                    return ((dt.hour + (dt.minute / 60) + (dt.second / 3600)) / 24).tolist()

                tweet_dt = pd.to_datetime(df.created_at).dt
                author_dt = pd.to_datetime(df.author_created_at).dt
                self.tweet_time.extend(get_normalised_time(tweet_dt))
                self.author_time.extend(get_normalised_time(author_dt))
                self.author_desc.extend([str(i) for i in df.author_description.tolist()])

        #classification: run clustering alg. to get cluster labels
        if self.classify:
            rads = np.radians(self.coords)
            self.clusterer = hdbscan.HDBSCAN(min_cluster_size=30, metric='haversine', algorithm='boruvka_kdtree', alpha=1.0, memory='./')
            self.cluster_labels = self.clusterer.fit_predict(rads)
            print('no. of clusters: ', self.clusterer.labels_.max())
        else:
            self.clusterer = None

    def __len__(self: Dataset) -> int:
        assert len(self.tweet_tokens) == len(self.coords)
        return len(self.tweet_tokens)

    def __getitem__(self: Dataset, idx: int):
        tokens = self.tweet_tokens[idx]
        if self.classify:
            labels = torch.FloatTensor(self.cluster_labels[idx])
        else:
            labels = torch.FloatTensor(self.coords[idx])
        metadata = None
        if self.use_metadata:
            # tweet_time = torch.nan_to_num(torch.FloatTensor([self.tweet_time[idx]]), 0.5)
            # author_time = torch.nan_to_num(torch.FloatTensor([self.author_time[idx]]), 0.5)
            fname = self.fnames[idx]
            tweet_time = torch.FloatTensor([self.tweet_time[idx]])
            author_time = torch.FloatTensor([self.author_time[idx]])
            author_desc = self.author_desc[idx]
            metadata = (tweet_time, author_time, author_desc, fname)

        return tokens, labels, metadata
