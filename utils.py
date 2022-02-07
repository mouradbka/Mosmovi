import math
import torch
import torch.nn.functional as F


EARTH_RADIUS = 6372.8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gc_distance(gold, pred):
    _degree_radian = lambda d: (d * math.pi) / 180
    rad_gold = _degree_radian(gold)
    rad_pred = _degree_radian(pred)

    cos_gold = torch.cos(rad_gold)
    sin_gold = torch.sin(rad_gold)
    cos_pred = torch.cos(rad_pred)
    sin_pred = torch.sin(rad_pred)

    n_gold = torch.stack([cos_gold[:, 0] * cos_gold[:, 1], cos_gold[:, 0] * sin_gold[:, 1], sin_gold[:, 0]], dim=1)
    n_pred = torch.stack([cos_pred[:, 0] * cos_pred[:, 1], cos_pred[:, 0] * sin_pred[:, 1], sin_pred[:, 0]], dim=1)

    return torch.acos(torch.inner(n_gold.to(device), n_pred.to(device)).diag()) * EARTH_RADIUS


def pad_chars(instance):
    chars, coords = zip(*instance)
    pad_length = max(map(len, chars))
    padded_chars = [F.pad(i, (0, pad_length - len(i)), value=255) for i in chars]
    return torch.stack(padded_chars), torch.stack(coords)


def train(batch, model, optimizer, criterion, device):
    chars, coords = batch
    pred = model(chars.to(device))

    optimizer.zero_grad()
    loss = criterion(pred, coords.to(device))
    loss.backward()
    optimizer.step()

    return loss


def evaluate(batch, model, criterion, device):
    chars, coords = batch
    pred = model(chars.to(device))
    loss = criterion(pred, coords.to(device))
    distance = gc_distance(coords, pred)

    return loss, distance


def split_users(df, percentage_users = 0.10):
    users_ids = df.author_id
    users_ids = set(df.author_id)
    n_samples = int(len(users_ids)*percentage_users)
    val_users = random.sample(users_ids, n_samples)
    return df.loc[df.author_id.isin(val_users)]
