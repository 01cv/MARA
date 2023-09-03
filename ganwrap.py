import os
import json
import random
import argparse
import datetime

import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from torchvision import utils, transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.cluster import KMeans


from encoder.fetch import fetch_encoder
from encoder.blackbox_encoder import BlackboxEncoder
from encoder.blackbox_encoder import WhiteboxEncoder
from generator.stylegan_utils import StyleGANWrapper
from dataset.parse_dataset import dataset_parser
from utils import cosine_similarity, str2bool



def load_model(args):
    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # encoder dict: contains all necessary stuff w.r.t. each encoders
    enc = fetch_encoder(args.encoder)
    enc = BlackboxEncoder(enc, img_size=enc.img_size).to(args.device)
    # fetch generator
    stgan = StyleGANWrapper(args)
    return stgan,enc





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_space', default='Wp', type=str, help='which latent space to use: [Z, W, Wp]')

    # target dataset & encoder
    parser.add_argument('--dataset', default='lfw', type=str, help='target dataset to attack')
    # parser.add_argument('--enc_tgt', default='VGGNet19', type=str, help='target encoder type')
    parser.add_argument('--encoder', default='VGGNet19', type=str,
                        help='encoders for measuring cosine similarity during optimization [FaceNet, VGGNet19, SwinTransformer]')

    # StyleGAN & alignment model parameters - need not change
    parser.add_argument('--resolution', default=256, type=int, help='StyleGAN output resolution')
    parser.add_argument('--batch_size', default=30, type=int, help='StyleGAN batch size. Reduce to avoid OOM')
    parser.add_argument('--truncation', default=0.8, type=int, help='interpolation weight w.r.t. initial latent')
    parser.add_argument('--generator_type', default='FFHQ-256', type=str)
    parser.add_argument('--crop_size', default=192, type=int, help='crop size for StyleGAN output')

    # Misc.
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--device_id', default=2, type=int, help='which gpu to use')
    parser.add_argument('--cosine_log_freq', default=10, type=int, help='how often to log cosine similarity')

    args = parser.parse_args()
    args.device = f'cuda:{args.device_id}'

    # reconstruct
    gan,enc = load_model(args)
    # print(gan,enc)
    # args.init_latent = gan.generator.avg_latent.clone()
    z = torch.randn(1, 512).to(args.device)
    w = gan.generator.style(z)
    init_image = gan(w)
    # print(init_image.shape)
    # PIL.Image.SAVE(x_G,f'results/tets.jpg')
    # x_G = w[0]
    utils.save_image(init_image.cpu(), f'results/1.jpg', nrow=1, normalize=True, range=(-1,1))
                                                                                                        # ndarr = init_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # im = Image.fromarray(ndarr)
    # im.save('results/tets.jpg')

    # utils.save_image(x_G, f'results/tets.jpg', nrow=1, normalize=True, range=(-1, 1))
    v_G = enc(init_image, flip=True)
    print(v_G)
    # loss = 1 - cosine_similarity(v_G, v_T).mean()