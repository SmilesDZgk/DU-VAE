from .flow import *
import torch
from torch import nn
import numpy as np
import math
from ..utils import log_sum_exp


class IAFEncoderBase(nn.Module):
    """docstring for EncoderBase"""

    def __init__(self):
        super(IAFEncoderBase, self).__init__()

    def sample(self, input, nsamples):
        """sampling from the encoder
        Returns: Tensor1, Tuple
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tuple: contains the tensor mu [batch, nz] and
                logvar[batch, nz]
        """

        z_T, log_q_z = self.forward(input, nsamples)

        return z_T, log_q_z

    def forward(self, x, n_sample):
        """
        Args:
            x: (batch_size, *)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        raise NotImplementedError

    def encode(self, input, args):
        """perform the encoding and compute the KL term

        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]

        """

        #  (batch, nsamples, nz)
        z_T, log_q_z = self.forward(input, args.nsamples)

        log_p_z = self.log_q_z_0(z=z_T)  # [b s nz]

        kl = log_q_z - log_p_z

        # free-bit
        if self.training and args.fb == 1 and args.target_kl > 0:
            kl_obj = torch.mean(kl, dim=[0, 1], keepdim=True)
            kl_obj = torch.clamp_min(kl_obj, args.target_kl)
            kl_obj = kl_obj.expand(kl.size(0), kl.size(1), -1)
            kl = kl_obj

        return z_T, kl.sum(dim=[1, 2])  # like KL

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
        # import ipdb
        # ipdb.set_trace()
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

        # if self.args.gamma <0:
        #     mu,logvar = self.trans_param(mu,logvar)

        # import ipdb
        # ipdb.set_trace()

        # (batch_size, 1, nz)
        mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
        var = logvar.exp()

        # (batch_size, nsamples, nz)
        dev = z - mu

        # (batch_size, nsamples)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        return log_density


class VariationalFlow(IAFEncoderBase):
    """Approximate posterior parameterized by a flow (https://arxiv.org/abs/1606.04934)."""

    def __init__(self, args, vocab_size, model_init, emb_init):
        super().__init__()

        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.args = args

        flow_depth = args.flow_depth
        flow_width = args.flow_width

        self.embed = nn.Embedding(vocab_size, args.ni)
        self.lstm = nn.LSTM(input_size=args.ni, hidden_size=args.enc_nh, num_layers=1,
                            batch_first=True, dropout=0)

        self.linear = nn.Linear(args.enc_nh, 4 * args.nz, bias=False)
        modules = []
        for _ in range(flow_depth):
            modules.append(InverseAutoregressiveFlow(num_input=args.nz,
                                                     num_hidden=flow_width * args.nz,  # hidden dim in MADE
                                                     num_context=2 * args.nz))
            modules.append(Reverse(args.nz))

        self.q_z_flow = FlowSequential(*modules)
        self.log_q_z_0 = NormalLogProb()
        self.softplus = nn.Softplus()
        self.reset_parameters(model_init, emb_init)

        self.BN = False
        if self.args.gamma > 0:
            self.BN = True
            self.mu_bn = nn.BatchNorm1d(args.nz, eps=1e-8)
            self.gamma = args.gamma
            nn.init.constant_(self.mu_bn.weight, self.args.gamma)
            nn.init.constant_(self.mu_bn.bias, 0.0)

        self.DP = False
        if self.args.p_drop > 0 and self.args.delta_rate > 0:
            self.DP = True
            self.p_drop = self.args.p_drop
            self.delta_rate = self.args.delta_rate

    def reset_parameters(self, model_init, emb_init):
        for name, param in self.lstm.named_parameters():
            # self.initializer(param)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
                # model_init(param)
            elif 'weight' in name:
                model_init(param)

        model_init(self.linear.weight)
        emb_init(self.embed.weight)

    def forward(self, input, n_samples):
        """Return sample of latent variable and log prob."""
        word_embed = self.embed(input)
        _, (last_state, last_cell) = self.lstm(word_embed)
        loc_scale, h = self.linear(last_state.squeeze(0)).chunk(2, -1)
        loc, scale_arg = loc_scale.chunk(2, -1)
        scale = self.softplus(scale_arg)

        if self.BN:
            ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
            #if ss < self.gamma:
            self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
            loc = self.mu_bn(loc)
        if self.DP and self.args.kl_weight >= self.args.drop_start:
            var = scale ** 2
            var = torch.dropout(var, p=self.p_drop, train=self.training)
            var += self.delta_rate * 1.0 / (2 * math.e * math.pi)
            scale = var ** 0.5

        loc = loc.unsqueeze(1)
        scale = scale.unsqueeze(1)
        h = h.unsqueeze(1)

        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z_0 = loc + scale * eps  # reparameterization
        log_q_z_0 = self.log_q_z_0(loc=loc, scale=scale, z=z_0)
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow) # [b s nz]

        if torch.sum(torch.isnan(z_T)):
            import ipdb
            ipdb.set_trace()

        ################
        if torch.rand(1).sum() <= 0.0005:
            if self.BN:
                self.mu_bn.weight

        return z_T, log_q_z
        # return z_0, log_q_z_0.sum(-1)

    def infer_param(self, input):
        word_embed = self.embed(input)
        _, (last_state, last_cell) = self.lstm(word_embed)
        loc_scale, h = self.linear(last_state.squeeze(0)).chunk(2, -1)
        loc, scale_arg = loc_scale.chunk(2, -1)
        scale = self.softplus(scale_arg)
        # logvar = scale_arg

        if self.BN:
            ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
            if ss < self.gamma:
                self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
            loc = self.mu_bn(loc)
        if self.DP and self.args.kl_weight >= self.args.drop_start:
            var = scale ** 2
            var = torch.dropout(var, p=self.p_drop, train=self.training)
            var += self.delta_rate * 1.0 / (2 * math.e * math.pi)
            scale = var ** 0.5

        return loc, torch.log(scale ** 2)

    def learn_feature(self, input):
        word_embed = self.embed(input)
        _, (last_state, last_cell) = self.lstm(word_embed)
        loc_scale, h = self.linear(last_state.squeeze(0)).chunk(2, -1)
        loc, scale_arg = loc_scale.chunk(2, -1)
        import ipdb
        ipdb.set_trace()
        if self.BN:
            loc = self.mu_bn(loc)
        loc = loc.unsqueeze(1)
        h = h.unsqueeze(1)
        z_T, log_q_z_flow = self.q_z_flow(loc, context=h)
        return loc, z_T


from .enc_resnet_v2 import ResNet


class FlowResNetEncoderV2(IAFEncoderBase):
    def __init__(self, args, ngpu=1):
        super(FlowResNetEncoderV2, self).__init__()
        self.ngpu = ngpu
        self.nz = args.nz
        self.nc = 1
        hidden_units = 512
        self.main = nn.Sequential(
            ResNet(self.nc, [64, 64, 64], [2, 2, 2]),
            nn.Conv2d(64, hidden_units, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_units),
            nn.ELU(),
        )
        self.linear = nn.Linear(hidden_units, 4 * self.nz)
        self.reset_parameters()
        self.delta_rate = args.delta_rate

        self.args = args

        flow_depth = args.flow_depth
        flow_width = args.flow_width

        modules = []
        for _ in range(flow_depth):
            modules.append(InverseAutoregressiveFlow(num_input=args.nz,
                                                     num_hidden=flow_width * args.nz,  # hidden dim in MADE
                                                     num_context=2 * args.nz))
            modules.append(Reverse(args.nz))

        self.q_z_flow = FlowSequential(*modules)
        self.log_q_z_0 = NormalLogProb()
        self.softplus = nn.Softplus()

        self.BN = False
        if self.args.gamma > 0:
            self.BN = True
            self.mu_bn = nn.BatchNorm1d(args.nz, eps=1e-8)
            self.gamma = args.gamma
            nn.init.constant_(self.mu_bn.weight, self.args.gamma)
            nn.init.constant_(self.mu_bn.bias, 0.0)

        self.DP = False
        if self.args.p_drop > 0 and self.args.delta_rate > 0:
            self.DP = True
            self.p_drop = self.args.p_drop
            self.delta_rate = self.args.delta_rate

    def reset_parameters(self):
        for m in self.main.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, input, n_samples):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = self.linear(output.view(output.size()[:2]))
        loc_scale, h = output.chunk(2, -1)
        loc, scale_arg = loc_scale.chunk(2, -1)
        scale = self.softplus(scale_arg)

        if self.BN:
            ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
            #if ss < self.gamma:
            self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
            loc = self.mu_bn(loc)

        if self.DP and self.args.kl_weight >= self.args.drop_start:
            var = scale ** 2
            var = torch.dropout(var, p=self.p_drop, train=self.training)
            var += self.delta_rate * 1.0 / (2 * math.e * math.pi)
            scale = var ** 0.5

        loc = loc.unsqueeze(1)
        scale = scale.unsqueeze(1)
        h = h.unsqueeze(1)

        eps = torch.randn((loc.shape[0], n_samples, loc.shape[-1]), device=loc.device)
        z_0 = loc + scale * eps  # reparameterization
        log_q_z_0 = self.log_q_z_0(loc=loc, scale=scale, z=z_0)
        z_T, log_q_z_flow = self.q_z_flow(z_0, context=h)
        log_q_z = (log_q_z_0 + log_q_z_flow)  # [b s nz]

        if torch.sum(torch.isnan(z_T)):
            import ipdb
            ipdb.set_trace()

        if torch.rand(1).sum() <= 0.001:
            if self.BN:
                self.mu_bn.weight

        return z_T, log_q_z


    def infer_param(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = self.linear(output.view(output.size()[:2]))
        loc_scale, h = output.chunk(2, -1)
        loc, scale_arg = loc_scale.chunk(2, -1)
        scale = self.softplus(scale_arg)

        if self.BN:
            ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
            if ss < self.gamma:
                self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
            loc = self.mu_bn(loc)
        if self.DP and self.args.kl_weight >= self.args.drop_start:
            var = scale ** 2
            var = torch.dropout(var, p=self.p_drop, train=self.training)
            var += self.delta_rate * 1.0 / (2 * math.e * math.pi)
            scale = var ** 0.5

        return loc, torch.log(scale ** 2)

    def learn_feature(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        output = self.linear(output.view(output.size()[:2]))
        loc_scale, h = output.chunk(2, -1)
        loc, _ = loc_scale.chunk(2, -1)
        if self.BN:
            ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
            if ss < self.gamma:
                self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
            loc = self.mu_bn(loc)
        loc = loc.unsqueeze(1)
        h = h.unsqueeze(1)
        z_T, log_q_z_flow = self.q_z_flow(loc, context=h)
        return loc, z_T


class NormalLogProb(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, loc=None, scale=None):
        if loc is None:
            loc = torch.zeros_like(z, device=z.device)
        if scale is None:
            scale = torch.ones_like(z, device=z.device)

        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)
