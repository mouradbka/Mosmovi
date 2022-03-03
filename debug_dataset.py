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
    fail_files = []
    fails = []
    tweet_dataset = TweetDataset(data_dir=args.data_dir, use_metadata=True)
    print(len(tweet_dataset))
    for i in tqdm.tqdm(tweet_dataset):
        if math.isnan(i[2][0]) or math.isnan(i[2][1]):
        # if times != times:
        #     fails.append(i[2][2])
            fail_files.append(i[2][-1])

    print(set(fail_files))


if __name__ == '__main__':
    main()
