# %%
import argparse
import logging
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.cuda import amp
from torch.nn.modules import activation
from torch.nn.modules.activation import Threshold
from tqdm import tqdm

from eval import eval_net

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision as tv
import utils
import models
from utils.dataset import BasicDataset
from utils.DriveDataset import DriveDataset
# %%
logger = logging.getLogger(__name__)

# dir_img = osp.join("..", "unet_dataset", "images", "trainval")
# dir_mask = osp.join("..", "unet_dataset", "labels", "trainval")

dir_img = osp.join(".", "dataset", "train")
dir_mask = osp.join(".", "dataset", "train_GT")

def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel)


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e',
                        '--epochs',
                        metavar='E',
                        type=int,
                        default=5,
                        help='Number of epochs',
                        dest='epochs')
    parser.add_argument('-b',
                        '--batch_size',
                        metavar='B',
                        type=int,
                        nargs='?',
                        default=1,
                        help='Batch size',
                        dest='batchsize')
    parser.add_argument('-l',
                        '--learning_rate',
                        metavar='LR',
                        type=float,
                        nargs='?',
                        default=0.0001,
                        help='Learning rate',
                        dest='lr')
    parser.add_argument('-f',
                        '--load',
                        dest='load',
                        default=False,
                        action='store_true',
                        help='Load model from a .pth file')
    parser.add_argument('-s',
                        '--scale',
                        dest='scale',
                        type=float,
                        default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v',
                        '--validation',
                        dest='val',
                        type=float,
                        default=0.1,
                        help='Percent of the data \
                              that is used as validation (0-100)')
    parser.add_argument('-d',
                        '--device',
                        default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='DDP parameter, do not modify')
    parser.add_argument('--model_type',
                        type=str,
                        default='utrans',
                        help="Model which choosed.")
    parser.add_argument('--split_seed', type=int, default=None, help='')
    return parser.parse_args()


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'UNetHX torch {torch.__version__} '
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ[
            'CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(
        ), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s)  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')

ROOT_FOLDER = "./DRIVE/"
task = "training" # "test"

def train_net(model,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_all_cp=True,
              dir_checkpoint='runs',
              split_seed=None):
    transform_valid = tv.transforms.Compose([tv.transforms.ToTensor(),
                                             tv.transforms.RandomCrop((400, 400))
                                             ])
    # dataset = BasicDataset(dir_img, dir_mask, transform=transform_valid)
    dataset = DriveDataset(task, ROOT_FOLDER, transform=transform_valid)

    n_val = int(len(dataset) *
                val_percent) if val_percent < 1 else int(val_percent)
    n_train = len(dataset) - n_val
    if split_seed:
        train, val = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(split_seed))
    else:
        train, val = random_split(dataset, [n_train, n_val])
    if type(model) == nn.parallel.DistributedDataParallel:
        train_loader = DataLoader(train,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True,
                                  sampler=DistributedSampler(train))
        val_loader = DataLoader(val,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=True,
                                sampler=DistributedSampler(val))
    else:
        train_loader = DataLoader(train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=True)
        val_loader = DataLoader(val,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=True)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_all_cp}
        Device:          {device.type}
    ''')

    # loss = nn.BCEWithLogitsLoss()
    # loss.__name__ = 'BCEWithLogitLoss'
    # loss = nn.BCELoss()
    # loss.__name__ = 'BCELoss'
    loss = utils.losses.NoiseRobustDiceLoss(eps=1e-7, activation='sigmoid')
    metrics = [
        utils.metrics.Dice(threshold=0.5, activation='sigmoid'),
        utils.metrics.Fscore(threshold=None, activation='sigmoid')
    ]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=lr),
    ])

    train_epoch = utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    valid_epoch = utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=True,
    )

    max_score = 0
    os.makedirs(dir_checkpoint, exist_ok=True)
    for i in range(0, epochs):
        print('\nEpoch: {}'.format(i + 1))
        #train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['dice_score']:
            max_score = valid_logs['dice_score']
            torch.save(model, osp.join(dir_checkpoint, 'best_model.pt'))
            torch.save(model.state_dict(),
                       osp.join(dir_checkpoint, 'best_model_dict.pth'))
            print('Model saved!')

        if save_all_cp:
            torch.save(model.state_dict(),
                       osp.join(dir_checkpoint, f'CP_epoch{i + 1}.pth'))
# BasicDataset
# def train_net(model,
#               device,
#               epochs=5,
#               batch_size=1,
#               lr=0.001,
#               val_percent=0.1,
#               save_all_cp=True,
#               dir_checkpoint='runs',
#               split_seed=None):
#     transform_valid = tv.transforms.Compose([tv.transforms.ToTensor(),
#                                              tv.transforms.RandomCrop((400, 400))
#                                              ])
#     dataset = BasicDataset(dir_img, dir_mask, transform=transform_valid)
#     n_val = int(len(dataset) *
#                 val_percent) if val_percent < 1 else int(val_percent)
#     n_train = len(dataset) - n_val
#     if split_seed:
#         train, val = random_split(
#             dataset, [n_train, n_val],
#             generator=torch.Generator().manual_seed(split_seed))
#     else:
#         train, val = random_split(dataset, [n_train, n_val])
#     if type(model) == nn.parallel.DistributedDataParallel:
#         train_loader = DataLoader(train,
#                                   batch_size=batch_size,
#                                   shuffle=False,
#                                   num_workers=0,
#                                   pin_memory=True,
#                                   sampler=DistributedSampler(train))
#         val_loader = DataLoader(val,
#                                 batch_size=batch_size,
#                                 shuffle=False,
#                                 num_workers=0,
#                                 pin_memory=True,
#                                 drop_last=True,
#                                 sampler=DistributedSampler(val))
#     else:
#         train_loader = DataLoader(train,
#                                   batch_size=batch_size,
#                                   shuffle=True,
#                                   num_workers=0,
#                                   pin_memory=True)
#         val_loader = DataLoader(val,
#                                 batch_size=batch_size,
#                                 shuffle=False,
#                                 num_workers=0,
#                                 pin_memory=True,
#                                 drop_last=True)

#     logging.info(f'''Starting training:
#         Epochs:          {epochs}
#         Batch size:      {batch_size}
#         Learning rate:   {lr}
#         Training size:   {n_train}
#         Validation size: {n_val}
#         Checkpoints:     {save_all_cp}
#         Device:          {device.type}
#     ''')

#     # loss = nn.BCEWithLogitsLoss()
#     # loss.__name__ = 'BCEWithLogitLoss'
#     # loss = nn.BCELoss()
#     # loss.__name__ = 'BCELoss'
#     loss = utils.losses.NoiseRobustDiceLoss(eps=1e-7, activation='sigmoid')
#     metrics = [
#         utils.metrics.Dice(threshold=0.5, activation='sigmoid'),
#         utils.metrics.Fscore(threshold=None, activation='sigmoid')
#     ]
#     optimizer = torch.optim.Adam([
#         dict(params=model.parameters(), lr=lr),
#     ])

#     train_epoch = utils.train.TrainEpoch(
#         model,
#         loss=loss,
#         metrics=metrics,
#         optimizer=optimizer,
#         device=device,
#         verbose=True,
#     )
#     valid_epoch = utils.train.ValidEpoch(
#         model,
#         loss=loss,
#         metrics=metrics,
#         device=device,
#         verbose=True,
#     )

#     max_score = 0
#     os.makedirs(dir_checkpoint, exist_ok=True)
#     for i in range(0, epochs):
#         print('\nEpoch: {}'.format(i + 1))
#         train_logs = train_epoch.run(train_loader)
#         valid_logs = valid_epoch.run(val_loader)

#         # do something (save model, change lr, etc.)
#         if max_score < valid_logs['dice_score']:
#             max_score = valid_logs['dice_score']
#             torch.save(model, osp.join(dir_checkpoint, 'best_model.pt'))
#             torch.save(model.state_dict(),
#                        osp.join(dir_checkpoint, 'best_model_dict.pth'))
#             print('Model saved!')

#         if save_all_cp:
#             torch.save(model.state_dict(),
#                        osp.join(dir_checkpoint, f'CP_epoch{i + 1}.pth'))
#         torch.cuda.empty_cache()

    
    
 


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = select_device(args.device, batch_size=args.batchsize)
    logging.info(f'Using device {device}')

    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    comment = f'MT_{args.model_type}_SS_{args.split_seed}_LR_{args.lr}_BS_{args.batchsize}'
    dir_checkpoint = osp.join(
        ".", "checkpoints",
        f"{current_time}_{socket.gethostname()}_" + comment)
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    nets = {
        # "unet": models.UNet,
        # "inunet": InUNet,
        # "attunet": AttU_Net,
        # "inattunet": InAttU_Net,
        # "att2uneta": Att2U_NetA,
        # "att2unetb": Att2U_NetB,
        # "att2unetc": Att2U_NetC,
        # "ecaunet": ECAU_Net,
        # "gsaunet": GsAUNet,
        # "utnet": U_Transformer,
        #"ddrnet": models.DualResNet,
        "utrans": models.U_Transformer
    }
    try:
        net_type = nets[args.model_type.lower()]
        net = net_type(in_channels=1, classes=1)
    except KeyError:
        os._exit(0)
    net.to(device=device)
    # net.apply(weight_init)

    cuda = device.type != 'cpu'
    # DP mode
    if cuda and args.local_rank == -1 and torch.cuda.device_count() > 1:
        print(f"DP Use multiple gpus: {args.device}")
        net = nn.DataParallel(net)
    # DDP mode
    if cuda and args.local_rank != -1 and torch.cuda.device_count() > 1:
        print(f"DDP Use multiple gpus: {args.device}")
        assert torch.cuda.device_count() > args.local_rank
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        net = nn.parallel.DistributedDataParallel(net)
    net = net.to(device=device)

    net = net.module if is_parallel(net) else net
    net = net.to(device=device)
    # logging.info(
    #     f'Network:\n'
    #     f'\t{net.n_channels} input channels\n'
    #     f'\t{net.n_classes} output channels (classes)\n'
    #     f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    train_net(model=net,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              device=device,
              val_percent=args.val,
              dir_checkpoint=dir_checkpoint,
              split_seed=args.split_seed,
              save_all_cp=True)

# %%
