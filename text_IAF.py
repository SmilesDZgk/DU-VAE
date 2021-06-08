import sys
import os
import time
import importlib
import argparse
import numpy as np
import torch
from torch import nn, optim
from data import MonoTextData, VocabEntry
from modules import VAEIAF as VAE
from modules import VariationalFlow, LSTMDecoder
from logger import Logger
from utils import calc_mi, calc_au

clip_grad = 5.0
decay_epoch = 5
lr_decay = 0.5
max_decay = 5


def init_config():
    parser = argparse.ArgumentParser(description='VAE-IAF mode collapse study')

    # model hyperparameters
    parser.add_argument('--dataset', default='synthetic', type=str, help='dataset to use')

    # optimization parameters
    parser.add_argument('--momentum', type=float, default=0, help='sgd momentum')
    parser.add_argument('--nsamples', type=int, default=1, help='number of samples for training')
    parser.add_argument('--iw_nsamples', type=int, default=500,
                        help='number of samples to compute importance weighted estimate')

    # select mode
    parser.add_argument('--eval', action='store_true', default=False, help='compute iw nll')
    parser.add_argument('--load_path', type=str, default='')

    # annealing paramters
    parser.add_argument('--warm_up', type=int, default=100, help="number of annealing epochs")
    parser.add_argument('--kl_start', type=float, default=1.0, help="starting KL weight")

    # these are for slurm purpose to save model
    parser.add_argument('--jobid', type=int, default=0, help='slurm job id')
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--delta_rate', type=float, default=1.0,
                        help=" coontrol the minization of the variation of latent variables")

    parser.add_argument('--gamma', type=float, default=0.8)  # BN-VAE

    parser.add_argument("--nz_new", type=int, default=32)

    parser.add_argument('--p_drop', type=float, default=0.3)  # p \in [0, 1]
    parser.add_argument('--lr', type=float, default=1.0)  # delta-VAE

    parser.add_argument('--flow_depth', type=int, default=2, help="depth of flow")
    parser.add_argument('--flow_width', type=int, default=2, help="width of flow")

    parser.add_argument("--fb", type=int, default=1,
                        help="0: no fb; 1: fb;")

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

    momentum_str = '_m%.2f' % args.momentum if args.momentum > 0 else ''
    id_ = "%s%s%s%s%s%s_dr%.2f_nz%d%s_%d_%d_%d%s_lr%.1f_IAF" % \
          (args.dataset, cw_str, load_str, gamma_str, fb_str, fd_str,
            args.delta_rate, args.nz_new, drop_str,
           args.jobid, args.taskid, args.seed, momentum_str, args.lr)
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

    if 'label' in params:
        args.label = params['label']
    else:
        args.label = False
    if 'vocab_file' not in params:
        args.vocab_file = None

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    return args


def test(model, test_data_batch, mode, args, verbose=True):
    report_kl_loss = report_kl_t_loss = report_rec_loss = 0
    report_num_words = report_num_sents = 0

    for i in np.random.permutation(len(test_data_batch)):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size

        loss, loss_rc, loss_kl = model.loss(batch_data, 1.0, args, training=False)
        loss_kl_t = model.KL(batch_data, args)
        assert (not loss_rc.requires_grad)
        assert (not loss_kl.requires_grad)

        loss_rc = loss_rc.sum()
        loss_kl = loss_kl.sum()
        loss_kl_t = loss_kl_t.sum()

        report_rec_loss += loss_rc.item()
        report_kl_loss += loss_kl.item()
        report_kl_t_loss += loss_kl_t.item()

    mutual_info = calc_mi(model, test_data_batch, device=args.device)

    test_loss = (report_rec_loss + report_kl_loss) / report_num_sents

    nll = (report_kl_t_loss + report_rec_loss) / report_num_sents
    kl = report_kl_loss / report_num_sents
    kl_t = report_kl_t_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)
    if verbose:
        print('%s --- avg_loss: %.4f, kl/H(z|x): %.4f, mi: %.4f, recon: %.4f, nll: %.4f, ppl: %.4f' % \
              (mode, test_loss, report_kl_t_loss / report_num_sents, mutual_info,
               report_rec_loss / report_num_sents, nll, ppl))
        sys.stdout.flush()

    return test_loss, nll, kl_t, ppl, mutual_info  # 返回真实的kl_t 不是训练中的kl


def calc_iwnll(model, test_data_batch, args, ns=100):
    report_nll_loss = 0
    report_num_words = report_num_sents = 0
    for id_, i in enumerate(np.random.permutation(len(test_data_batch))):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size
        if id_ % (round(len(test_data_batch) / 10)) == 0:
            print('iw nll computing %d0%%' % (id_ / (round(len(test_data_batch) / 10))))
            sys.stdout.flush()

        loss = model.nll_iw(batch_data, nsamples=args.iw_nsamples, ns=ns)

        report_nll_loss += loss.sum().item()

    nll = report_nll_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)

    print('iw nll: %.4f, iw ppl: %.4f' % (nll, ppl))
    sys.stdout.flush()
    return nll, ppl


def main(args):
    class uniform_initializer(object):
        def __init__(self, stdv):
            self.stdv = stdv

        def __call__(self, tensor):
            nn.init.uniform_(tensor, -self.stdv, self.stdv)

    if args.cuda:
        print('using cuda')

    print(args)

    opt_dict = {"not_improved": 0, "lr": args.lr, "best_loss": 1e4}

    if args.vocab_file is not None:
        print(args.vocab_file)
        vocab = {}
        with open(args.vocab_file) as fvocab:
            for i, line in enumerate(fvocab):
                vocab[line.strip()] = i
        vocab = VocabEntry(vocab)
        train_data = MonoTextData(args.train_data, label=args.label, vocab=vocab)
    else:
        train_data = MonoTextData(args.train_data, label=args.label)

    vocab = train_data.vocab

    vocab_size = len(vocab)

    val_data = MonoTextData(args.val_data, label=args.label, vocab=vocab)
    test_data = MonoTextData(args.test_data, label=args.label, vocab=vocab)

    print('Train data: %d samples' % len(train_data))
    print('finish reading datasets, vocab size is %d' % len(vocab))
    print('dropped sentences: %d' % train_data.dropped)
    sys.stdout.flush()

    log_niter = (len(train_data) // args.batch_size) // 10

    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)

    args.device = torch.device(args.device)
    device = args.device

    encoder = VariationalFlow(args, vocab_size, model_init, emb_init)
    args.enc_nh = args.dec_nh

    decoder = LSTMDecoder(args, vocab, model_init, emb_init)

    vae = VAE(encoder, decoder, args).to(device)

    if args.load_path:
        loaded_state_dict = torch.load(args.load_path, map_location=torch.device(device))
        vae.load_state_dict(loaded_state_dict, strict=False)
        print("%s loaded" % args.load_path)


    if args.eval:
        print('begin evaluation')
        vae.load_state_dict(torch.load(args.load_path, map_location=torch.device(device)))
        vae.eval()
        with torch.no_grad():
            test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                          device=device,
                                                          batch_first=True)
            test(vae, test_data_batch, "TEST", args)
            au, au_var = calc_au(vae, test_data_batch)
            print("%d active units" % au)
            # print(au_var)
            test_data_batch = test_data.create_data_batch(batch_size=1,
                                                          device=device,
                                                          batch_first=True)
            calc_iwnll(vae, test_data_batch, args)
        return

    enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=args.lr, momentum=args.momentum)
    dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=args.lr, momentum=args.momentum)
    opt_dict['lr'] = args.lr

    iter_ = decay_cnt = 0
    best_loss = 1e4
    vae.train()
    start = time.time()

    kl_weight = args.kl_start
    if args.warm_up > 0 and args.kl_start < 1.0:
        anneal_rate = (1.0 - args.kl_start) / (
                args.warm_up * (len(train_data) / args.batch_size))  # kl_start ==0 时 anneal_rate==0
    else:
        anneal_rate = 0

    train_data_batch = train_data.create_data_batch(batch_size=args.batch_size,
                                                    device=device,
                                                    batch_first=True)

    val_data_batch = val_data.create_data_batch(batch_size=args.batch_size,
                                                device=device,
                                                batch_first=True)

    test_data_batch = test_data.create_data_batch(batch_size=args.batch_size,
                                                  device=device,
                                                  batch_first=True)
    for epoch in range(args.epochs):
        report_kl_loss = report_rec_loss = 0
        report_num_words = report_num_sents = 0
        for i in np.random.permutation(len(train_data_batch)):  # len(train_data_batch)
            batch_data = train_data_batch[i]
            batch_size, sent_len = batch_data.size()
            if batch_data.size(0) < 16:
                continue

            # not predict start symbol
            report_num_words += (sent_len - 1) * batch_size

            report_num_sents += batch_size

            # kl_weight = 1.0
            if args.warm_up > 0 and args.kl_start < 1.0:
                kl_weight = min(1.0, kl_weight + anneal_rate)
            else:
                kl_weight = 1.0

            args.kl_weight = kl_weight

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            if args.fb == 0 or args.fb == 1:
                loss, loss_rc, loss_kl = vae.loss(batch_data, kl_weight, args)

            loss = loss.mean(dim=0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_grad)

            loss_rc = loss_rc.sum()
            loss_kl = loss_kl.sum()

            enc_optimizer.step()
            dec_optimizer.step()

            report_rec_loss += loss_rc.item()
            report_kl_loss += loss_kl.item()

            if iter_ % log_niter == 0:
                train_loss = (report_rec_loss + report_kl_loss) / report_num_sents
                if epoch == 0:
                    vae.eval()
                    with torch.no_grad():
                        mi = calc_mi(vae, val_data_batch, device=device)
                        au, _ = calc_au(vae, val_data_batch)
                    vae.train()

                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl/H(z|x): %.4f, mi: %.4f, recon: %.4f,' \
                          'au %d, time elapsed %.2fs' %
                          (epoch, iter_, train_loss, report_kl_loss / report_num_sents, mi,
                           report_rec_loss / report_num_sents, au, time.time() - start))
                else:
                    print('epoch: %d, iter: %d, avg_loss: %.4f, kl/H(z|x): %.4f, recon: %.4f,' \
                          'time elapsed %.2fs' %
                          (epoch, iter_, train_loss, report_kl_loss / report_num_sents,
                           report_rec_loss / report_num_sents, time.time() - start))

                sys.stdout.flush()

                report_rec_loss = report_kl_loss = 0
                report_num_words = report_num_sents = 0

            iter_ += 1

        print('kl weight %.4f' % args.kl_weight)

        vae.eval()
        with torch.no_grad():
            loss, nll, kl, ppl, mi = test(vae, val_data_batch, "VAL", args)
            au, au_var = calc_au(vae, val_data_batch)
            print("%d active units" % au)
            # print(au_var)

        if loss < best_loss:
            print('update best loss')
            best_loss = loss
            torch.save(vae.state_dict(), args.save_path)
        # torch.save(vae.state_dict(), args.save_path)
        if loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epoch >= 15:
                opt_dict["best_loss"] = loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                vae.load_state_dict(torch.load(args.save_path))
                print('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                enc_optimizer = optim.SGD(vae.encoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)
                dec_optimizer = optim.SGD(vae.decoder.parameters(), lr=opt_dict["lr"], momentum=args.momentum)

        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = loss

        if decay_cnt == max_decay:
            break

        if epoch % args.test_nepoch == 0:
            with torch.no_grad():
                loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)

        vae.train()

    # compute importance weighted estimate of log p(x)
    vae.load_state_dict(torch.load(args.save_path))

    vae.eval()
    with torch.no_grad():
        loss, nll, kl, ppl, _ = test(vae, test_data_batch, "TEST", args)
        au, au_var = calc_au(vae, test_data_batch)
        print("%d active units" % au)
        # print(au_var)

    test_data_batch = test_data.create_data_batch(batch_size=1,
                                                  device=device,
                                                  batch_first=True)
    with torch.no_grad():
        calc_iwnll(vae, test_data_batch, args)
    return args


if __name__ == '__main__':
    args = init_config()
    if not args.eval:
        sys.stdout = Logger(args.log_path)
    main(args)
