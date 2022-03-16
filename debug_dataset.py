import math
import tqdm

from argparse import ArgumentParser
from loader import TweetDataset
from models import *

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default='./data_full', action='store')
    args = parser.parse_args()
    n1, n2 = [], []
    tweet_dataset = TweetDataset(data_dir=args.data_dir, use_metadata=True)
    print(len(tweet_dataset))
    for i in tqdm.tqdm(tweet_dataset):
        n1.append(int(i[2][0].item() * 100))
        n2.append(int(i[2][1].item() * 100))

    print(np.mean(n1))
    print(np.mean(n2))
    print(set(n1))
    print(set(n2))


if __name__ == '__main__':
    main()
