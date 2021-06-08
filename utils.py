import numpy as np
import os, sys
import torch
from torch import nn, optim
import subprocess
from modules import VariationalFlow, FlowResNetEncoderV2
# from image_modules import GaussianEncoder
class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)



def calc_iwnll(model, test_data_batch, args, ns=100):
    report_nll_loss = 0
    report_num_words = report_num_sents = 0
    print("iw nll computing ", end="")
    for id_, i in enumerate(np.random.permutation(len(test_data_batch))):
        batch_data = test_data_batch[i]
        batch_size, sent_len = batch_data.size()

        # not predict start symbol
        report_num_words += (sent_len - 1) * batch_size

        report_num_sents += batch_size
        if id_ % (round(len(test_data_batch) / 20)) == 0:
            print('%d%% ' % (id_ / (round(len(test_data_batch) / 20)) * 5), end="")
            sys.stdout.flush()

        loss = model.nll_iw(batch_data, nsamples=args.iw_nsamples, ns=ns)

        report_nll_loss += loss.sum().item()

    print()
    sys.stdout.flush()

    nll = report_nll_loss / report_num_sents
    ppl = np.exp(nll * report_num_sents / report_num_words)

    return nll, ppl


def calc_mi(model, test_data_batch, device='cpu'):
    # calc_mi_v3
    import math
    from modules.utils import log_sum_exp

    mi = 0
    num_examples = 0

    mu_batch_list, logvar_batch_list = [], []
    neg_entropy = 0.
    for batch_data in test_data_batch:

        if isinstance(batch_data, list):
            batch_data = batch_data[0]

        batch_data = batch_data.to(device)

        if isinstance(model.encoder, VariationalFlow) \
                or isinstance(model.encoder, FlowResNetEncoderV2):
                # or isinstance(model.encoder, GaussianEncoder):
            mu, logvar = model.encode_stats(batch_data)
        else:
            mu, logvar = model.encoder.forward(batch_data)
        x_batch, nz = mu.size()
        ##print(x_batch, end=' ')
        num_examples += x_batch

        # E_{q(z|x)}log(q(z|x)) = -0.5*nz*log(2*\pi) - 0.5*(1+logvar).sum(-1)
        neg_entropy += (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).sum().item()
        mu_batch_list += [mu.cpu()]
        logvar_batch_list += [logvar.cpu()]

    neg_entropy = neg_entropy / num_examples

    num_examples = 0
    log_qz = 0.
    for i in range(len(mu_batch_list)):
        ###############
        # get z_samples
        ###############
        mu, logvar = mu_batch_list[i].to(device), logvar_batch_list[i].to(device)

        # [z_batch, 1, nz]
        if hasattr(model.encoder, 'reparameterize'):
            z_samples = model.encoder.reparameterize(mu, logvar, 1)
        else:
            z_samples = model.encoder.gaussian_enc.reparameterize(mu, logvar, 1)
        z_samples = z_samples.view(-1, 1, nz)
        num_examples += z_samples.size(0)

        ###############
        # compute density
        ###############
        # [1, x_batch, nz]

        indices = np.arange(len(mu_batch_list))
        mu = torch.cat([mu_batch_list[_] for _ in indices], dim=0).to(device)
        logvar = torch.cat([logvar_batch_list[_] for _ in indices], dim=0).to(device)
        x_batch, nz = mu.size()

        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z_samples - mu

        # (z_batch, x_batch)89
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz += (log_sum_exp(log_density, dim=1) - math.log(x_batch)).sum(-1)

    log_qz /= num_examples
    mi = neg_entropy - log_qz

    return mi


def calc_au(model, test_data_batch, delta=0.01):
    """compute the number of active units
    """
    cnt = 0
    for batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            means_sum = mean.sum(dim=0, keepdim=True)
        else:
            means_sum = means_sum + mean.sum(dim=0, keepdim=True)
        cnt += mean.size(0)

    # (1, nz)
    mean_mean = means_sum / cnt

    cnt = 0
    for batch_data in test_data_batch:
        mean, _ = model.encode_stats(batch_data)
        if cnt == 0:
            var_sum = ((mean - mean_mean) ** 2).sum(dim=0)
        else:
            var_sum = var_sum + ((mean - mean_mean) ** 2).sum(dim=0)
        cnt += mean.size(0)

    # (nz)
    au_var = var_sum / (cnt - 1)

    return (au_var >= delta).sum().item(), au_var


def sample_sentences(vae, vocab, device, num_sentences):
    global logging

    vae.eval()
    sampled_sents = []
    for i in range(num_sentences):
        z = vae.sample_from_prior(1)
        z = z.view(1, 1, -1)
        start = vocab.word2id['<s>']
        # START = torch.tensor([[[start]]])
        START = torch.tensor([[start]])
        end = vocab.word2id['</s>']
        START = START.to(device)
        z = z.to(device)
        vae.eval()
        sentence = vae.decoder.sample_text(START, z, end, device)
        decoded_sentence = vocab.decode_sentence(sentence)
        sampled_sents.append(decoded_sentence)
    for i, sent in enumerate(sampled_sents):
        logging(i, ":", ' '.join(sent))


def visualize_latent(args, epoch, vae, device, test_data):
    nsamples = 1

    with open(os.path.join(args.exp_dir, f'synthetic_latent_{epoch}.txt'), 'w') as f:
        test_data_batch, test_label_batch = test_data.create_data_batch_labels(batch_size=args.batch_size,
                                                                               device=device, batch_first=True)
        for i in range(len(test_data_batch)):
            batch_data = test_data_batch[i]
            batch_label = test_label_batch[i]
            batch_size, sent_len = batch_data.size()
            samples, _ = vae.encoder.encode(batch_data, nsamples)
            for i in range(batch_size):
                for j in range(nsamples):
                    sample = samples[i, j, :].cpu().detach().numpy().tolist()
                    f.write(batch_label[i] + '\t' + ' '.join([str(val) for val in sample]) + '\n')


cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}