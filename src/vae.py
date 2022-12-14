# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:37:37 2021

@author: bartm
"""
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import sys
import math 
import numpy as np
import pandas as pd
from torch.distributions.categorical import Categorical
from torch._six import string_classes
import collections
from torch.utils.data import Dataset, DataLoader


import pickle

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

import numpy as np
from collections import defaultdict


   
class VAE(nn.Module):
    def __init__(self, num_aa_type, dim_latent_vars, dim_msa_vars, num_hidden_units):
        super(VAE, self).__init__()

        ## num of amino acid types
        self.num_aa_type = num_aa_type

        ## dimension of latent space
        self.dim_latent_vars = dim_latent_vars

        ## dimension of binary representation of sequences
        self.dim_msa_vars = dim_msa_vars

        ## num of hidden neurons in encoder and decoder networks
        self.num_hidden_units = num_hidden_units

        ## encoder
        self.encoder_linears = nn.ModuleList()
        self.encoder_linears.append(nn.Linear(dim_msa_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.encoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.encoder_mu = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias = True)
        self.encoder_logsigma = nn.Linear(num_hidden_units[-1], dim_latent_vars, bias = True)

        ## decoder
        self.decoder_linears = nn.ModuleList()
        self.decoder_linears.append(nn.Linear(dim_latent_vars, num_hidden_units[0]))
        for i in range(1, len(num_hidden_units)):
            self.decoder_linears.append(nn.Linear(num_hidden_units[i-1], num_hidden_units[i]))
        self.decoder_linears.append(nn.Linear(num_hidden_units[-1], dim_msa_vars))

    def encoder(self, x):
        '''
        encoder transforms x into latent space z
        '''
        
        h = x
        for T in self.encoder_linears:
            h = T(h)
            h = torch.tanh(h)
        mu = self.encoder_mu(h)
        sigma = torch.exp(self.encoder_logsigma(h))
        return mu, sigma

    def decoder(self, z):
        '''
        decoder transforms latent space z into p, which is the log probability  of x being 1.
        '''
        
        h = z
        for i in range(len(self.decoder_linears)-1):
            h = self.decoder_linears[i](h)
            h = torch.tanh(h)
        h = self.decoder_linears[-1](h)

        fixed_shape = tuple(h.shape[0:-1])
        h = torch.unsqueeze(h, -1)
        h = h.view(fixed_shape + (-1, self.num_aa_type))
        
        #h = torch.reshape(h, fixed_shape + (-1, self.num_aa_type))        
        log_p = F.log_softmax(h, dim = -1)
        log_p = log_p.view(fixed_shape + (-1,))
        #log_p = torch.reshape(log_p, fixed_shape + (-1,))
        
        # h = h.view(h.size(0), -1, self.num_aa_type)
        # log_p = F.log_softmax(h, dim = 2)
        # log_p = log_p.view(log_p.size(0), -1)
        
        return log_p

    def compute_weighted_elbo(self, x, weight):
        ## sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        ## compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x*log_p, -1)

        ## compute elbo
        elbo = log_PxGz - torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        weight = weight / torch.sum(weight)
        elbo = torch.sum(elbo*weight)
        
        return elbo


    def compute_elbo(self, x, weight):
        ## sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps

        ## compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x*log_p, -1)

        ## compute elbo
        elbo = log_PxGz - torch.sum(0.5*(sigma**2 + mu**2 - 2*torch.log(sigma) - 1), -1)
        
        return elbo

    def compute_weighted_elbo(self, x, weight):
        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        # compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x * log_p, -1)

        # compute elbo
        elbo = log_PxGz - torch.sum(0.5 * (sigma**2 + mu**2 - 2 * torch.log(sigma) - 1), -1)
        weight = weight / torch.sum(weight)
        elbo = torch.sum(elbo * weight)

        return elbo

    def compute_p_importance_sampling(self, x, nsamples):

        with torch.no_grad():
            x = x.expand(nsamples, x.shape[0], x.shape[1])
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(-0.5 * z**2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x * log_p, -1)
            log_Pxz = log_Pz + log_PxGz

            log_QzGx = torch.sum(-0.5 * (eps)**2 -
                                 0.5 * torch.log(2 * z.new_tensor(np.pi))
                                 - torch.log(sigma), -1)        

            log_Px = torch.logsumexp(log_Pxz - log_QzGx, 0) - torch.log(torch.tensor(nsamples))

        return log_Px

    def compute_p(self, x, nsamples):

        with torch.no_grad():
            # sample z from prior
            z = torch.randn((x.shape[0], nsamples, self.dim_latent_vars), device=x.device)

            # compute log p(x|z)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x.unsqueeze(1).expand(-1, nsamples, -1) * log_p, -1)

        return torch.mean(log_PxGz, -1)


    def compute_elbo_no_grad(self, x):
        with torch.no_grad():
            # sample z from q(z|x)
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(sigma)
            z = mu + sigma * eps

            # compute log p(x|z)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x * log_p, -1)

            # compute elbo
            elbo = log_PxGz - torch.sum(0.5 * (sigma**2 + mu**2 - 2 * torch.log(sigma) - 1), -1)

        return elbo

    def sample(self, nsamples):

        # sample z from prior
        device = next(self.parameters()).device
        z = torch.randn((nsamples, self.dim_latent_vars), device=device)
        log_p = self.decoder(z)

        dist = Categorical(logits=log_p.reshape(nsamples, -1, self.num_aa_type))

        data = dist.sample()

        return data

    def compute_elbo(self, x):
        # sample z from q(z|x)
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(sigma)
        z = mu + sigma * eps

        # compute log p(x|z)
        log_p = self.decoder(z)
        log_PxGz = torch.sum(x * log_p, -1)

        # compute elbo
        elbo = log_PxGz - torch.sum(0.5 * (sigma**2 + mu**2 - 2 * torch.log(sigma) - 1), -1)

        return elbo

    def compute_elbo_with_multiple_samples(self, x, num_samples):
        with torch.no_grad():
            x = x.expand(num_samples, x.shape[0], x.shape[1])
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            log_Pz = torch.sum(-0.5 * z**2 - 0.5 * torch.log(2 * z.new_tensor(np.pi)), -1)
            log_p = self.decoder(z)
            log_PxGz = torch.sum(x * log_p, -1)
            log_Pxz = log_Pz + log_PxGz

            log_QzGx = torch.sum(-0.5 * (eps)**2 -
                                 0.5 * torch.log(2 * z.new_tensor(np.pi))
                                 - torch.log(sigma), -1)
            log_weight = (log_Pxz - log_QzGx).detach().data
            log_weight = log_weight.double()
            log_weight_max = torch.max(log_weight, 0)[0]
            log_weight = log_weight - log_weight_max
            weight = torch.exp(log_weight)
            elbo = torch.log(torch.mean(weight, 0)) + log_weight_max
            return elbo

    def sample_latent_var(self, mu, sigma):
        eps = torch.ones_like(sigma).normal_()
        z = mu + sigma * eps
        return z

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.sample_latent_var(mu, sigma)
        p = self.decoder(z)
        return mu, sigma, p


def vae_loss(model, batch):
    seq = batch[0]
    weights = batch[1]
    return -1*model.compute_weighted_elbo(seq, weights)