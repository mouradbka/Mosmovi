import numpy as np
import wandb
import utils
import tqdm

from argparse import ArgumentParser
from loader import TweetDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, ShuffleSplit
from models import *

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = ArgumentParser()
    # script
    parser.add_argument('--data_dir', default='./data', action='store')
    parser.add_argument('--arch', default='char_pool',
                        choices=['char_pool', 'char_lstm', 'char_cnn', 'char_lstm_cnn', 'char_transformer'])
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--save_prefix', default='model')
    # data
    parser.add_argument('--split_uids', action='store_true')
    parser.add_argument('--max_seq_len', default=-1)
    parser.add_argument('--subsample_ratio', default=-1)
    # model
    parser.add_argument('--loss', default='mse', choices=['mse', 'l1', 'smooth_l1'])
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'SGD'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project='mosmovi_1', config=args, name=args.run_name)

    tweet_dataset = TweetDataset(data_dir=args.data_dir)

    if args.split_uids:
        gss = GroupShuffleSplit(n_splits=1, train_size=0.9)
        train_indices, val_indices = next(gss.split(np.arange(len(tweet_dataset)), groups=tweet_dataset.uids))
    else:
        ss = ShuffleSplit(n_splits=1, train_size=0.9)
        train_indices, val_indices = next(ss.split(np.arange(len(tweet_dataset))))

    train_dataset = Subset(tweet_dataset, train_indices)
    val_dataset = Subset(tweet_dataset, val_indices)

    if args.subsample_ratio:
        train_dataset, val_dataset = utils.subsample_datasets(train_dataset, val_dataset, ratio=args.subsample_ratio)

    _train_iter = DataLoader(train_dataset, batch_size=int(args.batch_size), collate_fn=lambda x: utils.pad_chars(x, args.max_seq_len))
    _val_iter = DataLoader(val_dataset, batch_size=int(args.batch_size), collate_fn=lambda x: utils.pad_chars(x, args.max_seq_len))
    train_iter = tqdm.tqdm(_train_iter)
    val_iter = tqdm.tqdm(_val_iter)

    model_arch = utils.get_arch(args.arch)
    model = model_arch(args)
    model.to(device)

    criterion = utils.get_criterion(args.loss)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=float(args.lr))

    best_mean = 999999
    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_iter:
            train_loss = utils.train(batch, model, optimizer, criterion, device=device)
            train_iter.set_description(f"train loss: {train_loss.item()}")
            wandb.log({"train_loss": train_loss.item()})

        with torch.no_grad():
            distances = []
            for batch in val_iter:
                val_loss, val_distance = utils.evaluate(batch, model, criterion, device=device)
                val_iter.set_description(f"validation loss: {val_loss.item()}")
                wandb.log({"val_loss": val_loss.item(), "val_distance": torch.mean(val_distance)})
                distances.extend(val_distance.tolist())

            # log
            val_mean = np.mean(distances)
            val_median = np.median(distances)
            wandb.log({"val_mean": val_mean, "val_median": val_median})
            logger.info(f"val_mean: {val_mean}")
            logger.info(f"val_median: {val_median}")

            # save model
            is_best = val_mean < best_mean
            best_mean = min(val_mean, best_mean)

            if is_best:
                logger.warning(f"saving {args.save_prefix}.pt @ epoch {epoch}; mean distance: {val_mean}")
                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'criterion': criterion,
                    'epoch': epoch,
                    'arch': args.arch,
                    'val_mean': val_mean
                }, args.save_prefix + '.pt')


if __name__ == '__main__':
    main()
