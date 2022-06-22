from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import models
from losses import SemCKDLoss
from models.util import SelfAttention

import argparse
import os, sys
import time
import logging

from utils import *

parser = argparse.ArgumentParser(description='Self-Distillation CIFAR Training')
parser.add_argument('--seed', type=int, default=2, help='random seed')

# path
parser.add_argument('--dataroot', type=str, default='../data')
parser.add_argument('--saveroot', type=str, default='./results', help='models and logs are saved here')

# basic arguments
parser.add_argument('--model', default="cifarresnet18", type=str)
parser.add_argument('--dataset', default='cifar100', type=str, help="cifar100|cifar10")

# training hyperparameter
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--epoch', default=200, type=int, help="epoch for training")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_size', type=int, default=128, help="the sizeof batch")
parser.add_argument('--alpha', default=0.1, type=float)
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--lamda', default=3.0, type=float, help='cls loss weight ratio')
parser.add_argument('--decay', type=float, default=5e-4, help='weight decay')
# distill arguments
parser.add_argument('--temp', '-T', type=float, default=4.0)
parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_val = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

cudnn.benchmark = True

args.model_name = '{}_D_{}'.format(args.model, args.dataset)

# Data
print('==> Preparing dataset: {}'.format(args.dataset))

transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                      transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      transforms.Normalize((0.5071, 0.4865, 0.4409),
                                                           (0.2673, 0.2564, 0.2762))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
])

if args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root=args.dataroot,
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=args.dataroot,
        train=False,
        download=True,
        transform=transform_test
    )

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4
)

num_class = 100

# Model
print('==> Building model: {}'.format(args.model))
model = models.load_model(args.model, num_class)

logdir = os.path.join(args.saveroot, args.dataset + '_' + args.model,
                      'temp_' + str(args.temp) + '_lamda_' + str(args.lamda))
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')
logname = os.path.join(logdir, 'log.txt')

data = torch.rand(2, 3, 32, 32)
model.eval()
feat, out = model(data)
module_list = nn.ModuleList([])
module_list.append(model)
trainable_list = nn.ModuleList([])
trainable_list.append(model)

criterion_cls = nn.CrossEntropyLoss()

s_n = [f.shape[1] for f in feat[:-1]]
t_n = feat[-1].shape[1]
criterion_kd = SemCKDLoss(args)
self_attention = SelfAttention(args.feat_dim, s_n, t_n, args.temp)
module_list.append(self_attention)
trainable_list.append(self_attention)

criterion_list = nn.ModuleList([])
criterion_list.append(criterion_cls)  # classification loss
criterion_list.append(criterion_kd)  # other knowledge distillation loss

logger.info("args = %s", args)

logger.info('----------- Network Initialization --------------')
# logger.info('%s', model)
# logger.info('param_size = %fMB', count_parameters_in_MB(trainable_list))
# logger.info('----------------------------')

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    module_list.cuda()
    criterion_list.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')

if args.ngpu > 1:
    model = torch.nn.DataParallel(module_list, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

optimizer = optim.SGD(trainable_list.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)


def main():
    for epoch in range(start_epoch, args.epoch):
        weight = train(epoch, module_list)
        val_loss, val_acc = test(epoch, module_list)
        adjust_learning_rate(optimizer, epoch)

    print("Best Accuracy : {}".format(best_val))
    logger = logging.getLogger('best')
    logger.info('[Acc {:.3f}]'.format(best_val))


def train(epoch, module_list):
    print('\nEpoch: %d' % epoch)
    for module in module_list:
        module.train()
    model = module_list[0]
    train_loss = 0
    correct = 0
    total = 0
    train_kd_loss = 0

    end = time.time()
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        feat, outputs = model(inputs)

        # compute loss
        criterion_cls = criterion_list[0]
        criterion_kd = criterion_list[1]

        #   for deepest classifier
        loss = criterion_cls(outputs, labels)
        train_loss += loss.item()

        teacher_feature = feat[-1]
        s_value, f_target, weight = module_list[1](feat[0:-1], teacher_feature)
        loss_kd = criterion_kd(s_value, f_target, weight)

        loss += args.lamda * loss_kd
        train_kd_loss += loss_kd.item()

        _, predicted = torch.max(outputs, 1)
        # _, predicted_ensemble = torch.max(outputs[0]/2.0+torch.mul(outputs[1:],weight/2.0), 1)

        total += labels.size(0)
        correct += predicted.eq(labels.data).sum().float().cpu()
        # correct_ensemble += predicted_ensemble.eq(labels.data).sum().float().cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) | KD: %.3f '
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                        train_kd_loss / (batch_idx + 1)))

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [cls {:.3f}] [Acc {:.3f}]'.format(
        epoch, train_loss / (batch_idx + 1),
               train_kd_loss / (batch_idx + 1),
               100. * correct / total))

    return weight


def test(epoch, module_list):
    global best_val
    model = module_list[0]
    model.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    # Define a data loader for evaluating
    loader = testloader

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            _, outputs = model(inputs)
            criterion_cls = criterion_list[0]
            loss = torch.mean(criterion_cls(outputs, labels))

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum().float()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        val_loss / (batch_idx + 1),
        acc))

    if acc > best_val:
        best_val = acc
        checkpoint(acc, epoch)

    return val_loss / (batch_idx + 1), acc


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(logdir, 'ckpt.t7'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= args.epoch // 3:
        lr /= 10
    if epoch >= args.epoch * 2 // 3:
        lr /= 10
    if epoch >= args.epoch - 20:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
