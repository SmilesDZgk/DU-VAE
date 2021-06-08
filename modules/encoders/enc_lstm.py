import math
import torch
import torch.nn as nn
from .encoder import GaussianEncoderBase


class LSTMEncoder(GaussianEncoderBase):
    """Gaussian LSTM Encoder with constant-length batching"""

    def __init__(self, args, vocab_size, model_init, emb_init):
        super(LSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.args = args
        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)

        # dimension transformation to z (mean and logvar)
        self.linear = nn.Linear(args.enc_nh, 2 * args.nz, bias=False)

        self.reset_parameters(model_init, emb_init)
        self.delta_rate = args.delta_rate

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """
        word_embed = self.embed(input)
        _, (last_state, last_cell) = self.lstm(word_embed)
        mean, logvar = self.linear(last_state).chunk(2, -1)
        logvar = torch.log(torch.exp(logvar) + self.delta_rate * 1.0 / (2 * math.e * math.pi))

        return mean.squeeze(0), logvar.squeeze(0)


class GaussianLSTMEncoder(GaussianEncoderBase):
    """Gaussian LSTM Encoder with constant-length input"""

    def __init__(self, args, vocab_size, model_init, emb_init):
        super(GaussianLSTMEncoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.args = args

        self.embed = nn.Embedding(vocab_size, args.ni)

        self.lstm = nn.LSTM(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0)

        self.linear = nn.Linear(args.enc_nh, 2 * args.nz, bias=False)
        self.mu_bn = nn.BatchNorm1d(args.nz)
        self.gamma = args.gamma

        self.reset_parameters(model_init, emb_init)
        self.delta_rate = args.delta_rate

    def reset_parameters(self, model_init, emb_init, reset=False):
        if not reset:
            nn.init.constant_(self.mu_bn.weight, self.args.gamma)
        else:
            print('reset bn!')
            if self.args.gamma_train:
                nn.init.constant_(self.mu_bn.weight, self.args.gamma)
            else:
                self.mu_bn.weight.fill_(self.args.gamma)
            nn.init.constant_(self.mu_bn.bias, 0.0)

    def forward(self, input):
        """
        Args:
            x: (batch_size, seq_len)

        Returns: Tensor1, Tensor2
            Tensor1: the mean tensor, shape (batch, nz)
            Tensor2: the logvar tensor, shape (batch, nz)
        """

        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)

        _, (last_state, last_cell) = self.lstm(word_embed)
        mean, logvar = self.linear(last_state).chunk(2, -1)
        if self.args.gamma <= 0 or (mean.squeeze(0).size(0) == 1 and self.training == True):
            mean = mean.squeeze(0)
        else:
            self.mu_bn.weight.requires_grad = True
            ss = torch.mean(self.mu_bn.weight.data ** 2) ** 0.5
            #if ss < self.gamma:
            self.mu_bn.weight.data = self.mu_bn.weight.data * self.gamma / ss
            mean = self.mu_bn(mean.squeeze(0))

        if torch.sum(torch.isnan(mean)) or torch.sum(torch.isnan(logvar)):
            import ipdb
            ipdb.set_trace()

        if self.args.kl_weight == 1:
            logvar = torch.log(torch.exp(logvar) + self.delta_rate * 1.0 / (2 * math.e * math.pi))

        return mean, logvar.squeeze(0)

