import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical, Normal, OneHotCategorical
import numpy as np
import torch.nn.functional as F


def mdn_loss(y, pi, mu, sigma):
    g = gaussian_distribution(y, mu, sigma)
    nll = -torch.logsumexp(pi + g, dim=-1)
    return torch.mean(nll)


def gaussian_distribution(y,mu,sigma):
    y = y.unsqueeze(2).expand_as(mu)
    out = -torch.log(sigma) - (np.log(2 * np.pi) / 2) - 0.5 * (((y - mu) / sigma) ** 2)
    return out.sum(dim=1)


def sample(pi, mu, sigma):
    cat = Categorical(pi)
    ids = list(cat.sample().data)
    sampled = Variable(sigma.data.new(sigma.size(0), sigma.size(1)).normal_())
    for i, idx in enumerate(ids):
        sampled[i] = sampled[i].mul(sigma[i, :, idx]).add(mu[i, :, idx])
    return sampled


def predict(pi, mu, sigma, method='pi', top_k=False):
    if method == 'mixture':
        pis = pi.repeat(1, 2).view(pi.shape[0], 2, pi.shape[1])
        if top_k:
            indices = pis.topk(k=top_k, dim=-1).indices
            pis = torch.gather(pis, -1, indices)
            mu = torch.gather(mu, -1, indices)

        selected_mus = pis * mu
        return torch.sum(selected_mus, -1)

    elif method == 'pi':
        pis = torch.argmax(pi, axis=1)
        pis = pis.repeat(1, 2, 1).view(-1, 2, 1)
        selected_mus = torch.gather(mu, -1, pis)
        return selected_mus.squeeze(-1)


class MDN(nn.Module):
    def __init__(self, dim_in, dim_out, num_latent):
        super(MDN, self).__init__()
        self.dim_in = dim_in
        self.num_latent = num_latent
        self.dim_out = dim_out
        self.pi_h = nn.Linear(dim_in, num_latent)
        self.mu_h = nn.Linear(dim_in, dim_out * num_latent)
        self.sigma_h = nn.Linear(dim_in, dim_out * num_latent)
        self.elu = nn.ELU()

    def forward(self, x):
        pi = self.pi_h(x)
        pi = F.softmax(pi, dim=-1)
        mu = self.mu_h(x)
        mu = mu.view(-1, self.dim_out, self.num_latent)
        sigma = self.elu(self.sigma_h(x)) + 1
        sigma = sigma.view(-1, self.dim_out, self.num_latent)
        return pi, mu, sigma

