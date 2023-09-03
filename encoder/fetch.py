import os
import yaml
import torch
import sys
from encoder.VGGNets import VGG
from encoder.ResNets import Resnet
from encoder.Swin_Transformer import SwinTransformer
from AdaFace import net
from AdaFace.face_alignment import align
from termcolor import cprint
from collections import OrderedDict
import torch.nn as nn
from MagFace.models import iresnet
import argparse
from InsightFace_Pytorch import config
from InsightFace_Pytorch.model import Backbone, Arcface, MobileFaceNet

# from arcface.models.resnet import resnet_face18
# from arcface.models import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

cprint('=> parse the args ...', 'green')
parser = argparse.ArgumentParser(description='Trainer for posenet')
parser.add_argument('--arch', default='iresnet50', type=str,
                    help='backbone architechture')
parser.add_argument('--inf_list', default='', type=str,
                    help='the inference list')
parser.add_argument('--feat_list', type=str,
                    help='The save path for saveing features')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--embedding_size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--resume', default="/media/Storage1/Black-box-Face-Reconstruction-main1/MagFace/magface_iresnet50_MS1MV2_ddp_fp32.pth", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--cpu-mode', action='store_true', help='Use the CPU.')
parser.add_argument('--dist', default=1, help='use this if model is trained with dist')



args = parser.parse_args()


adaface_models = {
    'ir_50': "/media/Storage1/Black-box-Face-Reconstruction-main1/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
    # 'ir_101':"pretrained/adaface_ir101_ms1mv3.ckpt",
}

# adaface
def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location={'cuda:0': 'cuda:6'})['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict, False)
    model.eval()
    return model


# def load_model(model, model_path):
#     model_dict = model.state_dict()
#     pretrained_dict = torch.load(model_path)
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#     model.eval()
#     return model

# magface
def load_features(args):
    if args.arch == 'iresnet34':
        features = iresnet.iresnet34(
            pretrained=False,
            num_classes=args.embedding_size,
        )
    elif args.arch == 'iresnet18':
        features = iresnet.iresnet18(
            pretrained=False,
            num_classes=args.embedding_size,
        )
    elif args.arch == 'iresnet50':
        features = iresnet.iresnet50(
            pretrained=False,
            num_classes=args.embedding_size,
        )
    elif args.arch == 'iresnet100':
        features = iresnet.iresnet100(
            pretrained=False,
            num_classes=args.embedding_size,
        )
    else:
        raise ValueError()
    return features

def clean_dict_inf(model, state_dict):
    _state_dict = OrderedDict()
    for k, v in state_dict.items():
        # # assert k[0:1] == 'features.module.'
        new_k = 'features.'+'.'.join(k.split('.')[2:])
        if new_k in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_k].size():
            _state_dict[new_k] = v
        # assert k[0:1] == 'module.features.'
        new_kk = '.'.join(k.split('.')[1:])
        if new_kk in model.state_dict().keys() and \
           v.size() == model.state_dict()[new_kk].size():
            _state_dict[new_kk] = v
    num_model = len(model.state_dict().keys())
    num_ckpt = len(_state_dict.keys())
    if num_model != num_ckpt:
        sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
            num_model, num_ckpt))
    return _state_dict

class NetworkBuilder_inf(nn.Module):
    def __init__(self, args):
        super(NetworkBuilder_inf, self).__init__()
        self.features = load_features(args)

    def forward(self, input):
        # add Fp, a pose feature
        x = self.features(input)
        return x


def builder_inf(args):
    model = NetworkBuilder_inf(args)
    # Used to run inference
    model = load_dict_inf(args, model)
    return model
# model_recover = torch.load(args.model_recover_path, map_location={'cuda:0': 'cuda:2'})

def load_dict_inf(args, model):
    if os.path.isfile(args.resume):
        cprint('=> loading pth from {} ...'.format(args.resume))
        if args.cpu_mode:
            checkpoint = torch.load(args.resume, map_location=torch.device("cpu"))
        else:
            checkpoint = torch.load(args.resume, map_location={'cuda:0': 'cuda:6'})
        _state_dict = clean_dict_inf(model, checkpoint['state_dict'])
        model_dict = model.state_dict()
        model_dict.update(_state_dict)
        model.load_state_dict(model_dict)
        # delete to release more space
        del checkpoint
        del _state_dict
    else:
        sys.exit("=> No checkpoint found at '{}'".format(args.resume))
    return model
# magface

# arcface
def load_state(config):
    model = Backbone(config.get_config().net_depth, config.get_config().drop_ratio, config.get_config().net_mode).to(config.get_config().device)
    model.load_state_dict(torch.load("/media/Storage1/Black-box-Face-Reconstruction-main1/InsightFace_Pytorch/model_ir_se50.pth", map_location={'cuda:0': 'cuda:6'}))
    return model

def fetch_encoder(encoder_type, pretrained=True,
                  encoder_conf_file=f"{os.path.dirname(__file__)}/encoder_conf.yaml"):
    with open(encoder_conf_file) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        conf = conf[encoder_type]
    print('encoder param:', conf)

    if encoder_type == 'VGGNet19':
        encoder = VGG('VGG19')

    elif encoder_type == 'ResNet50':
        depth = conf['depth']  # depth of the ResNet, e.g. 50, 100, 152.
        drop_ratio = conf['drop_ratio']  # drop out ratio.
        net_mode = conf['net_mode']  # 'ir' for improved by resnt, 'ir_se' for SE-ResNet.
        feat_dim = conf['feat_dim']  # dimension of the output features, e.g. 512.
        out_h = conf['out_h']  # height of the feature map before the final features.
        out_w = conf['out_w']  # width of the feature map before the final features.
        encoder = Resnet(depth, drop_ratio, net_mode, feat_dim, out_h, out_w)

    elif encoder_type == 'SwinTransformer':
        img_size = conf['img_size']
        patch_size = conf['patch_size']
        in_chans = conf['in_chans']
        embed_dim = conf['embed_dim']
        depths = conf['depths']
        num_heads = conf['num_heads']
        window_size = conf['window_size']
        mlp_ratio = conf['mlp_ratio']
        drop_rate = conf['drop_rate']
        drop_path_rate = conf['drop_path_rate']
        encoder = SwinTransformer(img_size=img_size,
                                   patch_size=patch_size,
                                   in_chans=in_chans,
                                   embed_dim=embed_dim,
                                   depths=depths,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=True,
                                   qk_scale=None,
                                   drop_rate=drop_rate,
                                   drop_path_rate=drop_path_rate,
                                   ape=False,
                                   patch_norm=True,
                                   use_checkpoint=False)
    # elif encoder_type == 'FaceNet':
    #     from facenet_pytorch import InceptionResnetV1
    #     encoder = InceptionResnetV1(pretrained='vggface2')

    elif encoder_type == 'AdaFace':
        encoder = load_pretrained_model(architecture='ir_50')

    # elif encoder_type == 'ArcFace':
    #     encoder = load_model(resnet_face18(use_se=False),
    #                          "/media/ahu/Storage1/Black-box-Face-Reconstruction-main1/arcface/resnet18_110.pth")
    elif encoder_type == 'MagFace':
        encoder = builder_inf(args)

    elif encoder_type == 'ArcFace':
        encoder = load_state(config)


    else:
        raise NotImplementedError(f"{encoder_type} is not implemented!")

    # save image size & align info.
    encoder.align = conf['align']
    encoder.img_size = conf['img_size']

    # activate eval mode
    encoder.eval()

    if pretrained and encoder_type not in ['FaceNet', 'HOG']:
        stdict = torch.load(conf['weight'], map_location='cpu')
        encoder.load_state_dict(stdict,False)

    return encoder