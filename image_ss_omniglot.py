import os
import time
import importlib
import argparse
import sys
import numpy as np

import torch
from torch import nn, optim

from modules import ResNetEncoderV2,BNResNetEncoderV2, PixelCNNDecoderV2,FlowResNetEncoderV2
from modules import VAE, LinearDiscriminator_only
from logger import Logger
from omniglotDataset import Omniglot


# Junxian's new parameters
clip_grad = 1.0
decay_epoch = 2
lr_decay = 0.8
max_decay = 5


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--gamma_type', type=str, default='BN')
    parser.add_argument('--gamma_train', action="store_true", default=False)

    # model hyperparameters
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--dataset', default='omniglot', type=str, help='dataset to use')
    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd momentum')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="adam", help='sgd momentum')

    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                        help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str,
                        default='models/mnist/test/model.pt')  # TODO: 设定load_path

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=100,
                        help="number of annealing epochs. warm_up=0 means not anneal")
    parser.add_argument('--kl_start', type=float, default=1.0, help="starting KL weight")

    # output directory
    parser.add_argument('--exp_dir', default=None, type=str,
                        help='experiment directory.')
    parser.add_argument("--save_ckpt", type=int, default=0,
                        help="save checkpoint every epoch before this number")
    parser.add_argument("--save_latent", type=int, default=0)

    # new
    parser.add_argument("--reset_dec", action="store_true", default=True)
    parser.add_argument("--load_best_epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.)

    parser.add_argument("--fb", type=int, default=0,
                        help="0: no fb; 1: fb; E")
    parser.add_argument("--target_kl", type=float, default=-1,
                        help="target kl of the free bits trick")

    parser.add_argument("--batch_size", type=int, default=50,
                        help="number of epochs")
    parser.add_argument("--epochs", type=int, default=300,
                        help="number of epochs")
    parser.add_argument("--num_label", type=int, default=100,
                        help="t")
    parser.add_argument("--freeze_enc", action="store_true",
                        default=True)  # True-> freeze the parameters of vae.encoder
    parser.add_argument("--discriminator", type=str, default="linear")

    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--delta_rate', type=float, default=0.0,
                        help=" coontrol the minization of the variation of latent variables")

    parser.add_argument("--nz_new", type=int, default=32)  # myGaussianLSTMencoder

    parser.add_argument('--IAF', action='store_true', default=False)
    parser.add_argument('--flow_depth', type=int, default=2, help="depth of flow")
    parser.add_argument('--flow_width', type=int, default=60, help="width of flow")
    parser.add_argument('--p_drop', type=float, default=0)  # p \in [0, 1]

    args = parser.parse_args()

    # args.load_path ='models/omniglot/omniglot_aggressive1_KL1.00_dr0.00_beta-1.00_nz32_0_0_783435_betaF_4/model.pt'
    # args.load_path ='models/omniglot/omniglot_aggressive0_KL0.00_warm10_gamma0.50_BN_train5_dr1.00_beta-1.00_nz32_drop0.15_0_0_783435_betaF_5_large_de20/model.pt'
    # args.gamma = 0.5
    # args.gamma_type = 'BN'
    # args.load_path = 'models/omniglot/omniglot_fb2_tr0.20_fd2_fw60_dr0.00_nz32_0_0_783435_IAF/model.pt'
    # args.IAF = True

    # if len(args.load_path)>0:
    #     args.load_path = 'models/'+args.dataset+'/'+args.load_path

    # set args.cuda
    if 'cuda' in args.device:
        args.cuda = True
    else:
        args.cuda = False

    # set seeds
    seed_set = [783435, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    args.seed = seed_set[args.taskid]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    config_file = "config.config_%s_ss" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)

    load_str = "_load" if args.load_path != "" else ""
    opt_str = "_adam" if args.opt == "adam" else "_sgd"
    nlabel_str = "_nlabel{}".format(args.num_label)
    freeze_str = "_freeze" if args.freeze_enc else ""

    if len(args.load_path.split("/")) > 2:
        load_path_str = args.load_path.split("/")[2]
    else:
        load_path_str = args.load_path.split("/")[1]

    model_str = "_{}".format(args.discriminator)
    # set load and save paths
    if args.exp_dir == None:
        args.exp_dir = "models/exp_{}{}_ss_ft/{}{}{}{}{}".format(args.dataset,
                                                                 load_str, load_path_str, model_str, opt_str,
                                                                 nlabel_str, freeze_str)
    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    args.log_path = os.path.join(args.exp_dir, 'log.txt')

    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    args.kl_weight = 1

    return args


def test(model, test_loader, mode, args, verbose=False):

    report_correct = report_loss = 0
    report_num_sents = 0

    N=0
    for datum in test_loader:
        batch_data, batch_labels = datum
        batch_data = batch_data.to(args.device)
        batch_labels = batch_labels.to(args.device).squeeze()
        #batch_data = torch.bernoulli(batch_data)
        batch_size = batch_data.size(0)

        # not predict start symbol
        report_num_sents += batch_size
        loss, correct = model.get_performance_with_feature(batch_data, batch_labels)

        loss = loss.sum()

        report_loss += loss.item()
        report_correct += correct
        N+=1

    test_loss = report_loss / report_num_sents
    acc = report_correct / report_num_sents

    if verbose:
        print('%s --- avg_loss: %.4f, acc: %.4f' % \
                (mode, test_loss, acc))
        # sys.stdout.flush()

    return test_loss, acc

def train(args,dataset:Omniglot,task,device,trainum=10):
    x_train,l_train,x_test,l_test,NC = dataset.load_task(task,trainum)
    train_data = torch.utils.data.TensorDataset(x_train, l_train)
    test_data = torch.utils.data.TensorDataset(x_test, l_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    print('Train data: %d samples' % len(x_train))
    log_niter = max(1, (len(train_data) // (args.batch_size)) // 10)


    if args.discriminator == "linear":
        discriminator = LinearDiscriminator_only(args, NC).to(device)


    if args.opt == "sgd":
        optimizer = optim.SGD(discriminator.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == "adam":
        optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
        # optimizer = swats.SWATS(discriminator.parameters(), lr=0.001)

    else:
        raise ValueError("optimizer not supported")

    discriminator.train()

    iter_ = 0
    acc_loss = 0.
    for epoch in range(args.epochs):
        report_loss = 0
        report_correct = report_num_sents = 0
        acc_batch_size = 0
        optimizer.zero_grad()
        for datum in train_loader:
            batch_data, batch_labels = datum
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device).squeeze()
            batch_size = batch_data.size(0)
            if batch_data.size(0) < 2:
                continue

            # not predict start symbol
            report_num_sents += batch_size
            acc_batch_size += batch_size

            # (batch_size)
            loss, correct = discriminator.get_performance_with_feature(batch_data, batch_labels)

            acc_loss = loss.sum()
            acc_loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            report_loss += loss.sum().item()
            report_correct += correct
            iter_ += 1

        discriminator.eval()
        with torch.no_grad():
            loss, acc = test(discriminator, test_loader, "VAL", args, verbose=False)
        discriminator.train()

    # discriminator.load_state_dict(torch.load(args.save_path))
    discriminator.eval()
    with torch.no_grad():
        loss, acc = test(discriminator, test_loader, "TEST", args,verbose=True)
    return loss, acc,NC

def main(args):
    if args.cuda:
        print('using cuda')
    print(str(args))

    device = args.device


    if args.gamma > 0  and not args.IAF:
        encoder = BNResNetEncoderV2(args)
    elif not args.IAF:
        encoder = ResNetEncoderV2(args)
    elif args.IAF:
        encoder = FlowResNetEncoderV2(args)

    decoder = PixelCNNDecoderV2(args,mode='large')    # if args.HKL == 'H':

    vae = VAE(encoder, decoder, args).to(device)

    if args.load_path:
        loaded_state_dict = torch.load(args.load_path, map_location=torch.device(device))
        vae.load_state_dict(loaded_state_dict)
        print("%s loaded" % args.load_path)
    vae.eval()

    dataset = Omniglot(args.root, encoder=vae.encoder, device=device, IAF=args.IAF)

    for tasknum in [5,10,15]:
        acc_sum=0
        acc_wsum=0
        N=0
        acclist=[]
        NClist=[]
        for task in range(50):
            N+=1
            loss, acc,NC= train(args,dataset,task,device,tasknum)
            acc_sum+=acc
            acc_wsum+=NC*acc
            acclist.append(acc)
            NClist.append(NC)
        acc_mean = acc_sum/N
        acc_wmean = acc_wsum/sum(NClist)

        print('train_num', tasknum,'acc',acc_mean, acc_wmean)
        print(acclist)
    #     plt.plot(range(50),acclist,label = 'trainnum%d'%tasknum)
    # plt.show()
    # plt.legend()





if __name__ == '__main__':
    args = init_config()
    sys.stdout = Logger(args.log_path)
    print('---------------')
    main(args)
