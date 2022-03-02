import math
import tqdm

from argparse import ArgumentParser
from loader import TweetDataset
from models import *

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='./data', action='store')
    args = parser.parse_args()
    tweet_dataset = TweetDataset(data_dir=args.data_dir, use_metadata=True)
    print(len(tweet_dataset))
    for i in tqdm.tqdm(tweet_dataset):
        times = i[2][0]
        if math.isnan(times):
            print(i)


if __name__ == '__main__':
    main()
