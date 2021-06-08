import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LinearDiscriminator_only(nn.Module):
    """docstring for LinearDiscriminator"""

    def __init__(self, args, ncluster):
        super(LinearDiscriminator_only, self).__init__()
        self.args = args
        if args.IAF:
            self.linear = nn.Linear(args.nz, ncluster)
        else:
            self.linear = nn.Linear(args.nz, ncluster)
        self.loss = nn.CrossEntropyLoss(reduction="none")


    def get_performance_with_feature(self, batch_data, batch_labels):
        mu = batch_data
        logits = self.linear(mu)
        loss = self.loss(logits, batch_labels)

        _, pred = torch.max(logits, dim=1)
        correct = torch.eq(pred, batch_labels).float().sum().item()

        return loss, correct
