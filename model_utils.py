import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

ONEOVERSQRT2PI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians


class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

        self.elu = nn.ELU()

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        #sigma = torch.exp(self.sigma(minibatch))
        sigma = self.sigma(minibatch)
        sigma =  self.elu(sigma) + 1 + 1e-15
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return ret #torch.prod(ret, 2)

#def mdn_loss(pi, sigma, mu, target):
#    """Calculates the error, given the MoG parameters and the target
#    The loss is the negative log likelihood of the data given the MoG
#    parameters.
#    """
#    prob = pi * gaussian_probability(sigma, mu, target)
#    nll = -torch.log(torch.sum(prob, dim=1))
#    return torch.mean(nll)

def mdn_log_prob(pi, sigma, mu, y, temp=1):
    log_component_prob = gaussian_probability(sigma, mu, y)
    log_mix_prob = torch.log(
            nn.functional.gumbel_softmax(
                pi, tau=temp, dim=1) + 1e-15).unsqueeze(-1).repeat(1,1,2)
    return torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)

def mdn_loss(pi, sigma, mu, y):
    # NLL Loss
    log_prob = mdn_log_prob(pi, sigma, mu, y)
    loss = torch.mean(-log_prob)
    """
    if self.hparams.mdn_config.weight_regularization is not None:
        sigma_l1_reg = 0
        pi_l1_reg = 0
        mu_l1_reg = 0
        if self.hparams.mdn_config.lambda_sigma > 0:
            # Weight Regularization Sigma
            sigma_params = torch.cat(
                [x.view(-1) for x in self.mdn.sigma.parameters()]
            )
            sigma_l1_reg = self.hparams.mdn_config.lambda_sigma * torch.norm(
                sigma_params, self.hparams.mdn_config.weight_regularization
            )
        if self.hparams.mdn_config.lambda_pi > 0:
            pi_params = torch.cat([x.view(-1) for x in self.mdn.pi.parameters()])
            pi_l1_reg = self.hparams.mdn_config.lambda_pi * torch.norm(
                pi_params, self.hparams.mdn_config.weight_regularization
            )
        if self.hparams.mdn_config.lambda_mu > 0:
            mu_params = torch.cat([x.view(-1) for x in self.mdn.mu.parameters()])
            mu_l1_reg = self.hparams.mdn_config.lambda_mu * torch.norm(
                mu_params, self.hparams.mdn_config.weight_regularization
            )
    """
    loss = loss #+ sigma_l1_reg + pi_l1_reg + mu_l1_reg
    return loss



def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)


def sample(pi, sigma, mu):
    """Draw samples from a MoG."""
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)

    sample = Variable(sigma.data.new(sigma.size(0), 1).normal_())
    # Gathering from the n Gaussian Distribution based on sampled indices
    sample = sample * sigma.gather(1, pis) + mu.gather(1, pis)
    return sample

def generate_samples(pi, sigma, mu, n_samples=1000):
    samples = []
    softmax_pi = nn.functional.gumbel_softmax(
        pi, tau=1.0, dim=-1
    ).unsqueeze(-1).repeat(1,1,2)

    assert (
        softmax_pi < 0
    ).sum().item() == 0, "pi parameter should not have negative"

    for _ in range(n_samples):
        samples.append(sample(softmax_pi, sigma, mu))
    samples = torch.cat(samples, dim=1)
    return samples

def generate_point_predictions(self, pi, sigma, mu, n_samples=None):
    # Sample using n_samples and take average
    samples = self.generate_samples(pi, sigma, mu, n_samples)
    #if central_tendency == "mean":
    y_hat = torch.mean(samples, dim=-1)
    #elif central_tendency == "median":
    #y_hat = torch.median(samples, dim=-1).values
    return y_hat.unsqueeze(1)