import numpy as np
import wandb
import utils
import tqdm

from transformers import get_scheduler
from argparse import ArgumentParser
from loader import TweetDataset
from torch.utils.data import DataLoader, Subset
from transformers import BertTokenizer, ByT5Tokenizer
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, ShuffleSplit
from models import *

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = ArgumentParser()
    # script
    parser.add_argument('--data_dir', default='./data', action='store')
    parser.add_argument('--arch', default='char_pool',
                        choices=['char_pool', 'char_lstm', 'char_cnn', 'char_lstm_cnn',
                                 'bert', 'byt5'])
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--save_prefix', default='model')
    parser.add_argument('--use_metadata', action='store_true')
    # data
    parser.add_argument('--split_uids', action='store_true')
    parser.add_argument('--max_seq_len', default=-1, type=int)
    parser.add_argument('--subsample_ratio', default=-1)
    # model
    parser.add_argument('--loss', default='mse', choices=['mse', 'l1', 'smooth_l1'])
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--scheduler', default='constant', choices=['linear', 'constant'])
    parser.add_argument('--warmup_ratio', default=0.2, type=float)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--freeze_layers', type=int, default=0)
    parser.add_argument('--tweet_rbf_dim', type=int, default=50)
    parser.add_argument('--author_rbf_dim', type=int, default=10)
    parser.add_argument('--mdn', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project='mosmovi_1', config=args, name=args.run_name)

    tweet_dataset = TweetDataset(data_dir=args.data_dir, use_metadata=args.use_metadata)

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

    byte_tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-small')
    word_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', model_max_length=512)
    tokenizers = (byte_tokenizer, word_tokenizer)

    collate_fn = lambda instance: utils.pad_chars(instance, tokenizers, args.max_seq_len)
    _train_iter = DataLoader(train_dataset, batch_size=int(args.batch_size), collate_fn=collate_fn)
    _val_iter = DataLoader(val_dataset, batch_size=int(args.batch_size), collate_fn=collate_fn)
    train_iter = tqdm.tqdm(_train_iter)
    val_iter = tqdm.tqdm(_val_iter)

    model_arch = utils.get_arch(args.arch)
    model = model_arch(args)
    model.to(device)

    num_training_steps = args.num_epochs * len(train_iter)
    criterion = utils.get_criterion(args.loss)
    optimizer = utils.get_optimizer(args.optimizer)(model.parameters(), lr=args.lr)
    scheduler = get_scheduler(args.scheduler, optimizer, num_warmup_steps=num_training_steps * args.warmup_ratio,
                              num_training_steps=num_training_steps)

    best_mean = 999999
    for epoch in range(args.num_epochs):
        model.train()
        for i, batch in enumerate(train_iter):
            train_loss = utils.train(i, batch, model, optimizer, scheduler, criterion, args.gradient_accumulation_steps, args.mdn, device=device)
            train_iter.set_description(f"train loss: {train_loss.item()}")
            wandb.log({"train_loss": train_loss.item()})

        with torch.no_grad():
            model.eval()
            distances = []
            for batch in val_iter:
                val_loss, val_distance = utils.evaluate(batch, model, criterion, args.mdn, device=device)
                val_iter.set_description(f"validation loss: {val_loss.item()}")
                wandb.log({"val_loss": val_loss.item(), "val_distance": torch.mean(val_distance)})
                distances.extend(val_distance.tolist())

            # log
            val_mean = np.nan_to_num(np.mean(distances))
            val_median = np.nan_to_num(np.median(distances))
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
