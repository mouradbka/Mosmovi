import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical, Normal, OneHotCategorical
import numpy as np
import torch.nn.functional as F


def mdn_loss(y,pi,mu,sigma):

	"""
	# Parameters
    ----------
    y (batch_size x dim_out): vector of responses
    pi (batch_size x x num_latent) is priors
    mu (batch_size x dim_out x num_latent) is mean of each Gaussian
    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian
    Output
    ----------
    negative log likelihood loss
	"""

	g=gaussian_distribution(y,mu,sigma)
	print(g, ' gaus')
	prob=pi*g
	print(prob, ' prob')
	nll=-torch.log(torch.sum(prob,dim=-1))
	print(nll, ' nll')
	return torch.mean(nll)

def gaussian_distribution(y,mu,sigma):

	"""
	# Parameters
    ----------
    y (batch_size x dim_out): vector of responses
    mu (batch_size x dim_out x num_latent) is mean of each Gaussian
    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian
    Output
    ----------
	Gaussian distribution (batch_size x dim_out x num_latent)
	"""
	y=y.unsqueeze(2).expand_as(mu)
	one_div_sqrt_pi=1.0/np.sqrt(2.0*np.pi)

	x=(y.expand_as(mu)-mu)*torch.reciprocal(sigma)
	x=torch.exp(-0.5*x*x)*one_div_sqrt_pi
	x=x*torch.reciprocal(sigma)
	return torch.prod(x,1)


def sample(pi,mu,sigma):

	"""
	# Parameters
    ----------
    pi (batch_size x num_latent) is priors
    mu (batch_size x dim_out x num_latent) is mean of each Gaussian
    sigma (batch_size x dim_out x num_latent) is standard deviation of each Gaussian
    Output
    ----------
	"""
	cat=Categorical(pi)
	ids=list(cat.sample().data)
	sampled=Variable(sigma.data.new(sigma.size(0),
					sigma.size(1)).normal_())
	for i,idx in enumerate(ids):
		sampled[i]=sampled[i].mul(sigma[i,:,idx]).add(mu[i,:,idx])
	return sampled.data.numpy()


class MDN(nn.Module):
	def __init__(self,dim_in,dim_out,num_latent):
		super(MDN,self).__init__()
		self.dim_in=dim_in
		self.num_latent=num_latent
		self.dim_out=dim_out
		self.pi_h=nn.Linear(dim_in,num_latent)
		self.mu_h=nn.Linear(dim_in,dim_out*num_latent)
		self.sigma_h=nn.Linear(dim_in,dim_out*num_latent)
		self.elu = nn.ELU()

	def forward(self,x):
		pi=self.pi_h(x)
		pi=F.softmax(pi, dim=-1)
		mu=self.mu_h(x)
		mu=mu.view(-1,self.dim_out,self.num_latent)
		sigma=self.elu(torch.exp(self.sigma_h(x))) + 1
		sigma=sigma.view(-1,self.dim_out,self.num_latent)
		print(sigma)
		return pi,mu,sigma

