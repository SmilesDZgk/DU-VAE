import os
import time
import importlib
import argparse

import numpy as np

import torch
from torch import nn, optim
# import swats

from collections import defaultdict

from data import MonoTextData, VocabEntry
from modules import VAE,LinearDiscriminator_only
from modules import GaussianLSTMEncoder, LSTMEncoder, LSTMDecoder, VariationalFlow

from exp_utils import create_exp_dir
from utils import uniform_initializer

# old parameters
clip_grad = 5.0
decay_epoch = 2
lr_decay = 0.5
max_decay = 5

# Junxian's new parameters
# clip_grad = 1.0
# decay_epoch = 5
# lr_decay = 0.8
# max_decay = 10

logging = None


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')
    parser.add_argument('--gamma', type=float, default=0.0)
    # model hyperparameters
    parser.add_argument('--dataset', default='yelp', type=str, help='dataset to use')
    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--opt', type=str, choices=["sgd", "adam"], default="sgd", help='sgd momentum')

    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                        help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str,
                        default='short_yelp_aggressive0_hs1.00_warm100_0_0_783435.pt')  # TODO: 设定load_path

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

    parser.add_argument("--batch_size", type=int, default=16,
                        help="target kl of the free bits trick")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--update_every", type=int, default=1,
                        help="target kl of the free bits trick")
    parser.add_argument("--num_label", type=int, default=100,
                        help="target kl of the free bits trick")
    parser.add_argument("--freeze_enc", action="store_true",
                        default=True)  # True-> freeze the parameters of vae.encoder
    parser.add_argument("--discriminator", type=str, default="linear")

    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--delta_rate', type=float, default=0.0,
                        help=" coontrol the minization of the variation of latent variables")

    parser.add_argument('--p_drop', type=float, default=0)
    parser.add_argument('--IAF', action='store_true', default=False)
    parser.add_argument('--flow_depth', type=int, default=2, help="depth of flow")
    parser.add_argument('--flow_width', type=int, default=60, help="width of flow")

    args = parser.parse_args()

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

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    if args.num_label == 10:
        params = importlib.import_module(config_file).params_ss_10
    elif args.num_label == 100:
        params = importlib.import_module(config_file).params_ss_100
    elif args.num_label == 500:
        params = importlib.import_module(config_file).params_ss_500
    elif args.num_label == 1000:
        params = importlib.import_module(config_file).params_ss_1000
    elif args.num_label == 2000:
        params = importlib.import_module(config_file).params_ss_2000
    elif args.num_label == 10000:
        params = importlib.import_module(config_file).params_ss_10000

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
    if args.exp_dir is None:
        args.exp_dir = "models/exp_{}{}_ss_ft/{}{}{}{}{}".format(args.dataset,
                                                                 load_str, load_path_str, model_str, opt_str,
                                                                 nlabel_str, freeze_str)

    if len(args.load_path) <= 0 and args.eval:
        args.load_path = os.path.join(args.exp_dir, 'model.pt')

    args.save_path = os.path.join(args.exp_dir, 'model.pt')

    # set args.label
    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    args.kl_weight = 1

    return args


def test(model, test_data_batch, test_labels_batch, mode, args, verbose=True):
    global logging

    report_correct = report_loss = 0
    report_num_sents = 0
    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_labels = test_labels_batch[i]
        batch_labels = [int(x) for x in batch_labels]

        batch_labels = torch.tensor(batch_labels, dtype=torch.long, requires_grad=False, device=args.device)

        batch_size = batch_data.size(0)

        # not predict start symbol
        report_num_sents += batch_size

        loss, correct = model.get_performance_with_feature(batch_data, batch_labels)

        loss = loss.sum()

        report_loss += loss.item()
        report_correct += correct

    test_loss = report_loss / report_num_sents
    acc = report_correct / report_num_sents

    if verbose:
        logging('%s --- avg_loss: %.4f, acc: %.4f' % \
                (mode, test_loss, acc))
        # sys.stdout.flush()

    return test_loss, acc

def main(args):
    global logging
    logging = create_exp_dir(args.exp_dir, scripts_to_save=[])

    if args.cuda:
        logging('using cuda')
    logging(str(args))

    opt_dict = {"not_improved": 0, "lr": 1., "best_loss": 1e4}

    vocab = {}
    with open(args.vocab_file) as fvocab:
        for i, line in enumerate(fvocab):
            vocab[line.strip()] = i

    vocab = VocabEntry(vocab)

    train_data = MonoTextData(args.train_data, label=args.label, vocab=vocab)

    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)

    logging('Train data: %d samples' % len(train_data))
    logging('finish reading datasets, vocab size is %d' % len(vocab))
    logging('dropped sentences: %d' % train_data.dropped)
    # sys.stdout.flush()

    log_niter = max(1, (len(train_data) // (args.batch_size * args.update_every)) // 10)

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    # device = torch.device("cuda" if args.cuda else "cpu")
    # device = "cuda" if args.cuda else "cpu"
    device = args.device

    if args.gamma > 0 and args.enc_type == 'lstm' and not args.IAF:
        encoder = GaussianLSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    elif args.gamma == 0 and args.enc_type == 'lstm' and not args.IAF:
        encoder = LSTMEncoder(args, vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    elif args.IAF:
        encoder = VariationalFlow(args,vocab_size, model_init, emb_init)
        args.enc_nh = args.dec_nh
    else:
        raise ValueError("the specified encoder type is not supported")

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)

    vae = VAE(encoder, decoder, args).to(device)
    vae.eval()

    if args.load_path:
        loaded_state_dict = torch.load(args.load_path, map_location=torch.device(device))
        vae.load_state_dict(loaded_state_dict)
        logging("%s loaded" % args.load_path)

    try:
        print('theta', vae.encoder.theta)
    except:
        pass
    if args.discriminator == "linear":
        discriminator = LinearDiscriminator_only(args, args.ncluster).to(device)
    # elif args.discriminator == "mlp":
    #     discriminator = MLPDiscriminator(args, vae.encoder).to(device)

    if args.opt == "sgd":
        optimizer = optim.SGD(discriminator.parameters(), lr=args.lr, momentum=args.momentum)
        opt_dict['lr'] = args.lr
    elif args.opt == "adam":
        optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
        # optimizer = swats.SWATS(discriminator.parameters(), lr=0.001)
        opt_dict['lr'] = 0.001
    else:
        raise ValueError("optimizer not supported")

    iter_ = decay_cnt = 0
    best_loss = 1e4
    # best_kl = best_nll = best_ppl = 0
    # pre_mi = 0
    discriminator.train()
    start = time.time()

    train_data_batch, train_labels_batch = train_data.create_data_batch_labels(batch_size=args.batch_size,
                                                                               device=device,
                                                                               batch_first=True)

    val_data_batch, val_labels_batch = val_data.create_data_batch_labels(batch_size=128,
                                                                         device=device,
                                                                         batch_first=True)

    test_data_batch, test_labels_batch = test_data.create_data_batch_labels(batch_size=128,
                                                                            device=device,
                                                                            batch_first=True)
    #
    def learn_feature(data_batch,labels_batch):
        feature = []
        label = []
        for i in np.random.permutation(len(data_batch)):
            batch_data = data_batch[i]
            batch_labels = labels_batch[i]
            batch_data = batch_data.to(device)
            batch_size = batch_data.size(0)
            if args.IAF:
                loc, zT = vae.encoder.learn_feature(batch_data)
                # mu = torch.cat([loc, zT], dim=-1)
                mu=zT
                mu = mu.squeeze(1)
            else:
                mu, logvar = vae.encoder(batch_data)
            feature.append(mu.detach())
            label.append(batch_labels)
        return feature,label

    train_data_batch, train_labels_batch = learn_feature(train_data_batch, train_labels_batch)
    val_data_batch, val_labels_batch = learn_feature(val_data_batch, val_labels_batch)
    test_data_batch,test_labels_batch = learn_feature(test_data_batch,test_labels_batch)

    acc_cnt = 1
    acc_loss = 0.
    for epoch in range(args.epochs):
        report_loss = 0
        report_correct  = report_num_sents = 0
        acc_batch_size = 0
        optimizer.zero_grad()
        for i in np.random.permutation(len(train_data_batch)):
            batch_data = train_data_batch[i]
            if batch_data.size(0) < 2:
                continue
            batch_labels = train_labels_batch[i]
            batch_labels = [int(x) for x in batch_labels]

            batch_labels = torch.tensor(batch_labels, dtype=torch.long, requires_grad=False, device=device)

            batch_size = batch_data.size(0)

            # not predict start symbol
            report_num_sents += batch_size
            acc_batch_size += batch_size

            # (batch_size)
            loss, correct = discriminator.get_performance_with_feature(batch_data, batch_labels)

            acc_loss = acc_loss + loss.sum()

            if acc_cnt % args.update_every == 0:
                acc_loss = acc_loss / acc_batch_size
                acc_loss.backward()

                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), clip_grad)

                optimizer.step()
                optimizer.zero_grad()

                acc_cnt = 0
                acc_loss = 0
                acc_batch_size = 0

            acc_cnt += 1
            report_loss += loss.sum().item()
            report_correct += correct

            if iter_ % log_niter == 0:
                train_loss = report_loss / report_num_sents

            iter_ += 1

        # logging('lr {}'.format(opt_dict["lr"]))
        # print(report_num_sents)
        discriminator.eval()

        with torch.no_grad():
            loss, acc = test(discriminator, val_data_batch, val_labels_batch, "VAL", args)
            # print(au_var)

        if loss < best_loss:
            logging('update best loss')
            best_loss = loss
            best_acc = acc
            print(args.save_path)
            torch.save(discriminator.state_dict(), args.save_path)

        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epoch >= args.load_best_epoch:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                discriminator.load_state_dict(torch.load(args.save_path))
                logging('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                if args.opt == "sgd":
                    optimizer = optim.SGD(discriminator.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                    opt_dict['lr'] = opt_dict["lr"]
                elif args.opt == "adam":
                    optimizer = optim.Adam(discriminator.parameters(), lr=opt_dict["lr"])
                    opt_dict['lr'] = opt_dict["lr"]
                else:
                    raise ValueError("optimizer not supported")

        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            with torch.no_grad():
                loss, acc = test(discriminator, test_data_batch, test_labels_batch, "TEST", args)

        discriminator.train()

    # compute importance weighted estimate of log p(x)
    discriminator.load_state_dict(torch.load(args.save_path))
    discriminator.eval()

    with torch.no_grad():
        loss, acc = test(discriminator, test_data_batch, test_labels_batch, "TEST", args)
        # print(au_var)


if __name__ == '__main__':
    args = init_config()
    main(args)
