import numpy as np
import wandb
import utils
import tqdm
from collections import defaultdict

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
    parser.add_argument('--data_dir', default='./data_tiny', action='store')
    parser.add_argument('--arch', default='char_pool',
                        choices=['char_pool', 'char_lstm', 'char_cnn', 'char_lstm_cnn',
                                 'bert', 'byt5'])
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--save_prefix', default='model', help='Path to save the model to')
    parser.add_argument('--subsample', action='store_true', help='Use a maximum of 100k samples per location')
    # data
    parser.add_argument('--split_uids', action='store_true', help='Hold-out train user IDs when generating validation')
    parser.add_argument('--max_seq_len', default=-1, type=int, help='Truncate tweet text to length; -1 to disable')
    parser.add_argument('--subsample_ratio', default=-1, help='Subsample dataset to this ratio')
    # model
    parser.add_argument('--loss', default='mse', choices=['mse', 'l1', 'smooth_l1'])
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Number of training steps to accumulate gradient over')
    parser.add_argument('--scheduler', default='constant', choices=['linear', 'constant'],
                        help="`linear' scales up the LR over `args.warmup_ratio' steps")
    parser.add_argument('--warmup_ratio', default=0.2, type=float,
                        help="Ratio of maximum training steps to scale LR over")
    parser.add_argument('--lr', type=float, default=1e-4, help="Optimiser learning rate")
    parser.add_argument('--dropout', type=float, default=0.3, help="Model dropout ratio")
    parser.add_argument('--batch_size', type=int, default=128, help="Training batch size")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--freeze_layers', type=int, default=0,
                        help="Freeze bottom `n' layers when fine-tuning BERT/ByT5-based models")
    parser.add_argument('--reduce_layer', action='store_true', default=False, help="Add linear layer before output")
    parser.add_argument('--mdn', action='store_true', default=False,
                        help="Use Mixture Density Networks for confidence estimation")
    parser.add_argument('--reg_penalty', type=float, default=0.0, help="Regularise MDN mean/variance parameters")
    parser.add_argument('--entropy_loss_weight', type=float, default=0.0,
                        help="Apply entropy loss to distribution over MDN means")
    parser.add_argument('--num_confidence_bins', type=int, default=5, help="Number of confidence levels")
    parser.add_argument('--entropy_confidence', action='store_true', default=False,
                        help="Use entropy to estimate confidence (default: max. probabilities)")
    parser.add_argument('--num_gaussians', type=int, default=10,
                        help="Number of Gaussian distributions used by the MDN")
    parser.add_argument('--confidence_validation_criterion', action='store_true', default=False,
                        help="Save models based on score of the highest confidence level (default: average over all validation points)")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project='mosmovi_1', config=args, name=args.run_name)
    tweet_dataset = TweetDataset(data_dir=args.data_dir, subsample=args.subsample)

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
    _train_iter = DataLoader(train_dataset, batch_size=int(args.batch_size), collate_fn=collate_fn, shuffle=True)
    _val_iter = DataLoader(val_dataset, batch_size=int(args.batch_size), collate_fn=collate_fn, shuffle=True)
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
        for batch in enumerate(train_iter):
            train_loss = utils.train(batch, model, optimizer, scheduler, criterion, args, device=device)
            train_iter.set_description(f"train loss: {train_loss.item()}")
            wandb.log({"train_loss": train_loss.item()})

        with torch.no_grad():
            model.eval()
            distances = []
            distances_confidence = defaultdict(list)
            for batch in val_iter:

                eval_stats = utils.evaluate(batch, model, criterion, args, device=device)
                if args.mdn:
                    val_loss, (val_distance, val_distance_confidence) = eval_stats
                    for confidence_level, corresp_val_distance in val_distance_confidence.items():
                        distances_confidence[int(confidence_level)].extend(corresp_val_distance.tolist())

                else:
                    val_loss, val_distance = eval_stats

                val_iter.set_description(f"validation loss: {val_loss.item()}")
                wandb.log({"val_loss": val_loss.item(), "val_distance": torch.mean(val_distance)})
                distances.extend(val_distance.tolist())

            # log
            val_mean = np.nan_to_num(np.mean(distances))
            val_median = np.nan_to_num(np.median(distances))
            wandb.log({"val_mean": val_mean, "val_median": val_median})
            logger.info(f"val_mean: {val_mean}")
            logger.info(f"val_median: {val_median}")

            if args.mdn:
                for confidence_level, corresp_val_distance_list in distances_confidence.items():
                    val_mean_c = np.nan_to_num(np.mean(corresp_val_distance_list))
                    val_median_c = np.nan_to_num(np.median(corresp_val_distance_list))
                    wandb.log({"conf: " + str(confidence_level) + " - " + "val_mean": val_mean_c,
                               "conf: " + str(confidence_level) + " - " + "val_median": val_median_c})
                    logger.info(f"conf: " + str(confidence_level) + " - " + f"val_mean: {val_mean_c}")
                    logger.info(f"conf: " + str(confidence_level) + " - " + f"val_median: {val_median_c}")

            # save model, use highest conf prediction scores if flag on
            if args.mdn and args.confidence_validation_criterion:
                val_mean = val_mean_c
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
