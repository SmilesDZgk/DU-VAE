import sys
import os
import time
import importlib
import argparse

import numpy as np

import torch
import torch.utils.data
# from torchvision.utils import save_image
from torch import nn, optim

from modules import FlowResNetEncoderV2, PixelCNNDecoderV2
from modules import VAEIAF as VAE
from logger import Logger
from utils import calc_mi

clip_grad = 5.0
decay_epoch = 20
lr_decay = 0.5
max_decay = 5


def init_config():
    parser = argparse.ArgumentParser(description='VAE mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', default='omniglot', type=str, help='dataset to use')

    # optimization parameters
    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                        help='number of samples to compute importance weighted estimate')
    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')
    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=10)
    parser.add_argument('--kl_start', type=float, default=1.0)
    # these are for slurm purpose to save model
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--delta_rate', type=float, default=1.0,
                        help=" coontrol the minization of the variation of latent variables")
    parser.add_argument('--gamma', type=float, default=0.5)  # BN-VAE
    parser.add_argument("--reset_dec", action="store_true", default=False)
    parser.add_argument("--nz_new", type=int, default=32)  # myGaussianLSTMencoder
    parser.add_argument('--p_drop', type=float, default=0.15)  # p \in [0, 1]

    parser.add_argument('--flow_depth', type=int, default=2, help="depth of flow")
    parser.add_argument('--flow_width', type=int, default=2, help="width of flow")

    parser.add_argument("--fb", type=int, default=0,
                        help="0: no fb; 1: ")

    parser.add_argument("--target_kl", type=float, default=0.0,
                        help="target kl of the free bits trick")
    parser.add_argument('--drop_start', type=float, default=1.0, help="starting KL weight")

    args = parser.parse_args()
    if 'cuda' in args.device:
        args.cuda = True
    else:
        args.cuda = False

    load_str = "_load" if args.load_path != "" else ""
    save_dir = "models/%s%s/" % (args.dataset, load_str)

    if args.warm_up > 0 and args.kl_start < 1.0:
        cw_str = '_warm%d' % args.warm_up + '_%.2f' % args.kl_start
    else:
        cw_str = ''

    if args.fb == 0:
        fb_str = ""
    elif args.fb in [1, 2]:
        fb_str = "_fb%d_tr%.2f" % (args.fb, args.target_kl)

    else:
        fb_str = ''

    drop_str = '_drop%.2f' % args.p_drop if args.p_drop != 0 else ''
    if 1.0 > args.drop_start > 0 and args.p_drop != 0:
        drop_str += '_start%.2f' % args.drop_start

    seed_set = [783435, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    args.seed = seed_set[args.taskid]

    if args.gamma > 0:
        gamma_str = '_gamma%.2f' % (args.gamma)

    else:
        gamma_str = ''

    if args.flow_depth > 0:
        fd_str = '_fd%d_fw%d' % (args.flow_depth, args.flow_width)

    id_ = "%s%s%s%s%s%s_dr%.2f_nz%d%s_%d_%d_%d_IAF" % \
          (args.dataset, cw_str, load_str, gamma_str, fb_str, fd_str,
            args.delta_rate, args.nz_new, drop_str,
           args.jobid, args.taskid, args.seed)
    save_dir += id_
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, 'model.pt')

    args.save_path = save_path
    print("save path", args.save_path)

    args.log_path = os.path.join(save_dir, "log.txt")
    print("log path", args.log_path)

    # load config file into args
    config_file = "config.config_%s" % args.dataset
    params = importlib.import_module(config_file).params
    args = argparse.Namespace(**vars(args), **params)
    if args.nz != args.nz_new:
        args.nz = args.nz_new
        print('args.nz', args.nz)

    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    return args


def test(model, test_loader, mode, args):
    report_kl_loss = report_kl_t_loss = report_rec_loss = 0
    report_num_examples = 0
    mutual_info = []
    for datum in test_loader:
        batch_data, _ = datum
        batch_data = batch_data.to(args.device)

        batch_size = batch_data.size(0)

        report_num_examples += batch_size

        loss, loss_rc, loss_kl = model.loss(batch_data, 1.0, args, training=False)
        loss_kl_t = model.KL(batch_data, args)

        assert (not loss_rc.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss_kl_t = loss_kl_t.sum()

        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()
        report_kl_t_loss += loss_kl_t.item()

    mutual_info = calc_mi(model, test_loader, device=args.device)

    test_loss = (report_rec_loss + report_kl_loss) / report_num_examples

    nll = (report_kl_t_loss + report_rec_loss) / report_num_examples
    kl = report_kl_loss / report_num_examples
    kl_t = report_kl_t_loss / report_num_examples

    print('%s --- avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f, nll: %.4f' % \
          (mode, test_loss, report_kl_t_loss / report_num_examples, mutual_info,
           report_rec_loss / report_num_examples, nll))
    sys.stdout.flush()

    return test_loss, nll, kl_t  ##返回真实的kl_t 不是训练中的kl


def calc_iwnll(model, test_loader, args):
    report_nll_loss = 0
    report_num_examples = 0
    for id_, datum in enumerate(test_loader):
        batch_data, _ = datum
        batch_data = batch_data.to(args.device)

        batch_size = batch_data.size(0)

        report_num_examples += batch_size

        if id_ % (round(len(test_loader) / 10)) == 0:
            print('iw nll computing %d0%%' % (id_ / (round(len(test_loader) / 10))))
            sys.stdout.flush()

        loss = model.nll_iw(batch_data, nsamples=args.iw_nsamples)

        report_nll_loss += loss.sum().item()

    nll = report_nll_loss / report_num_examples

    print('iw nll: %.4f' % nll)
    sys.stdout.flush()
    return nll

def calc_au(model, test_loader, delta=0.01):
    """compute the number of active units
    """
    means = []
    for datum in test_loader:
        batch_data, _ = datum

        batch_data = batch_data.to(args.device)

        mean, _ = model.encode_stats(batch_data)
        means.append(mean)

    means = torch.cat(means, dim=0)
    au_mean = means.mean(0, keepdim=True)

    # (batch_size, nz)
    au_var = means - au_mean
    ns = au_var.size(0)

    au_var = (au_var ** 2).sum(dim=0) / (ns - 1)

    return (au_var >= delta).sum().item(), au_var

def main(args):
    if args.cuda:
        print('using cuda')
    print(args)

    args.device = torch.device(args.device)
    device = args.device

    opt_dict = {"not_improved": 0, "lr": 0.001, "best_loss": 1e4}

    all_data = torch.load(args.data_file)
    x_train, x_val, x_test = all_data
    if args.dataset == 'omniglot':
        x_train = x_train.to(device)
        x_val = x_val.to(device)
        x_test = x_test.to(device)
        y_size = 1
        y_train = x_train.new_zeros(x_train.size(0), y_size)
        y_val = x_train.new_zeros(x_val.size(0), y_size)
        y_test = x_train.new_zeros(x_test.size(0), y_size)

        print(torch.__version__)
        train_data = torch.utils.data.TensorDataset(x_train, y_train)
        val_data = torch.utils.data.TensorDataset(x_val, y_val)
        test_data = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)
    print('Train data: %d batches' % len(train_loader))
    print('Val data: %d batches' % len(val_loader))
    print('Test data: %d batches' % len(test_loader))
    sys.stdout.flush()

    log_niter = len(train_loader) // 5

    encoder = FlowResNetEncoderV2(args)
    decoder = PixelCNNDecoderV2(args)

    vae = VAE(encoder, decoder, args).to(device)


    if args.eval:
        print('begin evaluation')
        args.kl_weight = 1
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True)
        vae.load_state_dict(torch.load(args.load_path))
        vae.eval()
        with torch.no_grad():
            test(vae, test_loader, "TEST", args)
            au, au_var = calc_au(vae, test_loader)
            print("%d active units" % au)
            # print(au_var)
            calc_iwnll(vae, test_loader, args)
        return

    enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=0.001)
    dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=0.001)
    opt_dict['lr'] = 0.001

    iter_ = 0
    best_loss = 1e4
    best_kl = best_nll = best_ppl = 0
    decay_cnt = pre_mi = best_mi = mi_not_improved = 0
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    anneal_rate = (1.0 - args.kl_start) / (args.warm_up * len(train_loader))

    for epoch in range(args.epochs):

        report_kl_loss = report_rec_loss = 0
        report_num_examples = 0
        for datum in train_loader:
            batch_data, _ = datum
            batch_data = batch_data.to(device)
            batch_data = torch.bernoulli(batch_data)
            batch_size = batch_data.size(0)
            report_num_examples += batch_size

            kl_weight = min(1.0, kl_weight + anneal_rate)
            args.kl_weight = kl_weight

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, args)

            loss = loss.mean(dim=-1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

            loss_rc = loss_rc.sum()
            loss_kl = loss_kl.sum()

            enc_optimizer.step()
            dec_optimizer.step()

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            if iter_ % log_niter == 0:

                train_loss = (report_rec_loss + report_kl_loss) / report_num_examples
                if epoch == 0:
                    vae.eval()
                    with torch.no_grad():
                        mi = calc_mi(vae, val_loader, device=device)
                        au, _ = calc_au(vae, val_loader)

                    vae.train()

                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, mi: %.4f, recon: %.4f,' \
                          'au %d, time elapsed %.2fs' %
                          (epoch, iter_, train_loss, report_kl_loss / report_num_examples, mi,
                           report_rec_loss / report_num_examples, au, time.time() - start))
                else:
                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl: %.4f, recon: %.4f,' \
                          'time elapsed %.2fs' %
                          (epoch, iter_, train_loss, report_kl_loss / report_num_examples,
                           report_rec_loss / report_num_examples, time.time() - start))
                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_examples = 0

            iter_ += 1



        print('kl weight %.4f' % args.kl_weight)
        print('epoch: %d, VAL' % epoch)

        vae.eval()

        with torch.no_grad():
            loss, nll, kl = test(vae, val_loader, "VAL", args)
            au, au_var = calc_au(vae, val_loader)
            print("%d active units" % au)
            # print(au_var)

        if loss < best_loss:
            print('update best loss')
            best_loss = loss
            best_nll = nll
            best_kl = kl
            torch.save(vae.state_dict(), args.save_path)

        if loss > best_loss:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                vae.load_state_dict(torch.load(args.save_path))
                decay_cnt += 1
                print('new lr: %f' % opt_dict["lr"])
                enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=opt_dict["lr"])
                dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=opt_dict["lr"])
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            with torch.no_grad():
                loss, nll, kl = test(vae, test_loader, "TEST", args)

        vae.train()

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))
    vae.eval()
    with torch.no_grad():
        loss, nll, kl = test(vae, test_loader, "TEST", args)
        au, au_var = calc_au(vae, test_loader)
        print("%d active units" % au)
        # print(au_var)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=50, shuffle=True)

    with torch.no_grad():
        calc_iwnll(vae, test_loader, args)


if __name__ == '__main__':
    args = init_config()
    if  not args.eval:
        sys.stdout = Logger(args.log_path)
    main(args)
