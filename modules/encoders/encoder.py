import math
import torch
import torch.nn as nn
import random
from ..utils import log_sum_exp


class GaussianEncoderBase(nn.Module):
    """docstring for EncoderBase"""

    def __init__(self):
        super(GaussianEncoderBase, self).__init__()

    def forward(self, x):
        """
        Args:
            x: (batch_size, *)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        raise NotImplementedError

    def sample(self, input, nsamples):
        """sampling from the encoder
        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: contains the tensor mu [batch, nz] and
                logvar[batch, nz]
        """

        # (batch_size, nz)
        mu, logvar = self.forward(input)
        # if self.args.gamma<0:
        #     mu, logvar = self.trans_param(mu, logvar)

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)
        # if self.args.gamma <0:
        #     z=self.z_bn(z.squeeze(1)).unsqueeze(1)

        return z, (mu, logvar)

    def encode(self, input, args, training=True):
        """perform the encoding and compute the KL term

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]

        """
        nsamples = args.nsamples
        # (batch_size, nz)
        mu, logvar = self.forward(input)

        if args.p_drop > 0 and training and args.kl_weight == 1:
            var = logvar.exp() - args.delta_rate * 1.0 / (2 * math.e * math.pi)
            var = torch.dropout(var, p=args.p_drop, train=training)
            logvar = torch.log(var + args.delta_rate * 1.0 / (2 * math.e * math.pi))

        # (batch, nsamples, nz)
        z = self.reparameterize(mu, logvar, nsamples)

        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        KL = KL.sum(dim=1) #B

        # if torch.sum(torch.isnan(HKL)):
        #     import ipdb
        #     ipdb.set_trace()

        return z, KL

    def MPD(self,mu,logvar):
        eps = 1e-9
        z_shape = [mu.size(1)]
        batch_size = mu.size(0)
        # [batch, z_shape]
        var = logvar.exp()
        # B [batch, batch, z_shape]
        B = (mu.unsqueeze(1) - mu.unsqueeze(0)).pow(2).div(var.unsqueeze(0) + eps)
        B = B.mean(0).mean(0) #z_shape
        A = var.mean(0)
        C = (1/(var+eps)).mean(0)
        # if torch.max(var)>10:
        #     import ipdb
        #     ipdb.set_trace()
        return 0.5*(B+A*C-1)

    def CE(self, logvar):
        return 0.5*(logvar.mean(dim=0) + math.log(2*math.pi*math.e))

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)

            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)

        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """

        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)

    def eval_inference_dist(self, x, z, param=None):
        """this function computes log q(z | x)
        Args:
            z: tensor
                different z points that will be evaluated, with
                shape [batch, nsamples, nz]
        Returns: Tensor1
            Tensor1: log q(z|x) with shape [batch, nsamples]
        """

        nz = z.size(2)

        if not param:
            mu, logvar = self.forward(x)
        else:
            mu, logvar = param

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density

    def calc_mi(self, x):
        """Approximate the mutual information between x and z
        I(x, z) = E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        Returns: Float

        """

        # [x_batch, nz]
        mu, logvar = self.forward(x)

        # if self.args.gamma<0:
        #     mu, logvar = self.trans_param( mu, logvar)

        x_batch, nz = mu.size()

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z_samples = self.reparameterize(mu, logvar, 1)

        # [1, x_batch, nz]
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()





