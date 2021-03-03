'''
 * Copyright (c) 2018, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import argparse
import os
import random
import shutil
import time
import warnings
import builtins
import json
from datetime import datetime
import sys
import math
import numpy as np

import tensorboard_logger
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

import loader
from Model import Model
from resnet import *

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',choices=['resnet50','resnet50x2','resnet50x4'])
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=160, type=int,
                    help='supervised batch size')
parser.add_argument('--batch-size-u', default=640, type=int,
                    help='unsupervised batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--cos', default=True, help='use cosine lr schedule')
parser.add_argument('--schedule', default=[], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to pretrained model (default: none)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True)

## CoMatch settings
parser.add_argument('--temperature', default=0.1, type=float, help='temperature for similarity scaling')
parser.add_argument('--low-dim', default=128, type=int, help='feature dimension')
parser.add_argument('--moco-m', default=0.996, type=float,
                    help='momentum of updating momentum encoder')
parser.add_argument('--K', default=30000, type=int, help='size of memory bank and momentum queue')
parser.add_argument('--thr', default=0.6, type=float, help='pseudo-label confidence threshold')
parser.add_argument('--contrast-th', default=0.3, type=float, help='pseudo-label graph connection threshold')
parser.add_argument('--lam-u', default=10, type=float, help='weight for unsupervised cross-entropy loss')
parser.add_argument('--lam-c', default=10, type=float, help='weight for unsupervised contrastive loss')
parser.add_argument('--alpha', default=0.9, type=float, help='weight for model prediction in constructing pseudo-label')
parser.add_argument('--exp_dir', default='experiment/comatch_1percent', type=str, help='experiment directory')

## dataset settings
parser.add_argument('--percent', type=int, default=1, choices=[1,10], help='percentage of labeled samples')
parser.add_argument('--num-class', default=1000, type=int)
parser.add_argument('--annotation', default='annotation_1percent.json', type=str, help='annotation file')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
        
    os.makedirs(args.exp_dir, exist_ok=True)
        
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
        
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet50':
        model = Model(resnet50,args,width=1)
    elif args.arch == 'resnet50x2':    
        model = Model(resnet50,args,width=2)
    elif args.arch == 'resnet50x4':    
        model = Model(resnet50,args,width=4)
    else:
        raise NotImplementedError('model not supported {}'.format(args.arch))    
    
    # load moco-v2 pretrained model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q'):
                    # remove prefix
                    state_dict[k.replace('module.encoder_q', 'encoder')] = state_dict[k]     
                # delete renamed or unused k
                del state_dict[k]
            for k in list(state_dict.keys()):
                if 'fc.0' in k:
                    state_dict[k.replace('fc.0','fc1')] = state_dict[k]
                if 'fc.2' in k:
                    state_dict[k.replace('fc.2','fc2')] = state_dict[k]            
                    del state_dict[k]   
            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))            
            # copy paramter to the momentum encoder
            for param, param_m in zip(model.encoder.parameters(), model.m_encoder.parameters()):
                param_m.data.copy_(param.data)  
                param_m.requires_grad = False                
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
            
    
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_u = int(args.batch_size_u / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]) #find_unused_parameters=True
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criteria_x = nn.CrossEntropyLoss().cuda(args.gpu)    
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True
                               )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    print("=> preparing dataset")
    # Data loading code         
    
    transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),                
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) 
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),                         
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    transform_weak = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),                                     
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])    
    transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    
    three_crops_transform = loader.ThreeCropsTransform(transform_weak, transform_strong, transform_strong)
    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
 
    labeled_dataset = datasets.ImageFolder(traindir,transform_weak)
    unlabeled_dataset = datasets.ImageFolder(traindir,three_crops_transform)

    if not os.path.exists(args.annotation):
        # randomly sample labeled data on main process (gpu0)
        label_per_class = 13 if args.percent==1 else 128
        if args.gpu==0:
            random.shuffle(labeled_dataset.samples)
            labeled_samples=[]
            unlabeled_samples=[]
            num_img = torch.zeros(args.num_class)

            for i,(img,label) in enumerate(labeled_dataset.samples):
                if num_img[label]<label_per_class:
                    labeled_samples.append((img,label))
                    num_img[label]+=1
                else:
                    unlabeled_samples.append((img,label))        
            annotation = {'labeled_samples':labeled_samples,'unlabeled_samples':unlabeled_samples}
            with open(args.annotation,'w') as f:
                json.dump(annotation,f)
            print('save annotation to %s'%args.annotation)   
        dist.barrier()
    print('load annotation from %s'%args.annotation)
    annotation = json.load(open(args.annotation,'r'))
    
    if args.percent==1:
        # repeat labeled samples for faster dataloading
        labeled_dataset.samples = annotation['labeled_samples']*10
    else:
        labeled_dataset.samples = annotation['labeled_samples']
    unlabeled_dataset.samples = annotation['unlabeled_samples']
    
    labeled_sampler = torch.utils.data.distributed.DistributedSampler(labeled_dataset)
    labeled_loader = torch.utils.data.DataLoader(
        labeled_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=labeled_sampler)    
 
    unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled_dataset)
    unlabeled_loader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=int(args.batch_size_u), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=unlabeled_sampler)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transform_eval),
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # create loggers
    if args.gpu==0:
        tb_logger = tensorboard_logger.Logger(logdir=os.path.join(args.exp_dir,'tensorboard'), flush_secs=2)
        logger = setup_default_logging(args)
        logger.info(dict(args._get_kwargs()))        
    else:
        tb_logger = None
        logger = None

    for epoch in range(args.start_epoch, args.epochs):
        if epoch==0:
            args.m = 0.99 # larger update in first epoch
        else:
            args.m = args.moco_m
       
        adjust_learning_rate(optimizer, epoch, args)

        train(labeled_loader, unlabeled_loader, model, criteria_x, optimizer, epoch, args, logger, tb_logger)

        # evaluate on validation set
        acc1 = validate(val_loader, model, args, logger, tb_logger, epoch)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            },filename='{}/checkpoint_{:04d}.pth.tar'.format(args.exp_dir,epoch))
    
    # evaluate ema model
    acc1 = validate(val_loader, model, args, logger, tb_logger, -1)
    
            
def train(labeled_loader, unlabeled_loader, model, criteria_x, optimizer, epoch, args, logger, tb_logger):
    unlabeled_loader.sampler.set_epoch(epoch)  
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    loss_contrast_meter = AverageMeter()
    # the number of confident unlabeled samples that are correctly-predicted 
    n_correct_u_lbs_meter = AverageMeter()
    # the number of unlabeled samples with confident pseudo-labels
    n_conf = AverageMeter()
    # the number of positives (edges) in the pseudo-label graph
    pos_meter = AverageMeter()
    
    # switch to train mode
    model.train()
    
    labeled_loader.sampler.set_epoch(epoch*len(unlabeled_loader))
    iter_labeled_loader = iter(labeled_loader)
    end = time.time()   
    for i, unlabeled_batch in enumerate(unlabeled_loader):
        try:
            labeled_batch = next(iter_labeled_loader)
        except StopIteration:
            labeled_loader.sampler.set_epoch(epoch*len(unlabeled_loader)+i+1)
            iter_labeled_loader = iter(labeled_loader)
            labeled_batch = next(iter_labeled_loader)
                
        lbs_u_real = unlabeled_batch[1].cuda(args.gpu, non_blocking=True)  
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        # probs: soft pseudo-label
        # Q: pseudo-label graph
        # sim: embedding graph
        outputs_x, outputs_u_s0, lbs_x, probs, Q, sim = model(args,labeled_batch,unlabeled_batch,epoch=epoch)

        loss_x = criteria_x(outputs_x, lbs_x)
        
        with torch.no_grad():         
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args.thr).float()
        
        # unsupervised cross-entropy              
        loss_u = - torch.sum((F.log_softmax(outputs_u_s0,dim=1) * probs),dim=1) * mask
        loss_u = loss_u.mean()
      
        # remove edges with low similairty and normalize pseudo-label graph   
        pos_mask = (Q>=args.contrast_th)       
        Q_mask = Q * pos_mask
        Q_mask = Q_mask / Q_mask.sum(1,keepdim=True)
        
        positives = sim * pos_mask
        pos_probs = positives / sim.sum(1, keepdim=True)       
        log_probs = torch.log(pos_probs + 1e-7) * pos_mask
        
        # unsupervised contrastive loss   
        loss_contrast = - (log_probs*Q_mask).sum(1)
        loss_contrast = loss_contrast.mean()
        
        # ramp up the weight for unsupervised contrastive loss (optional)
        lam_c = min(epoch+1, args.lam_c)
        loss = loss_x + args.lam_u * loss_u + lam_c * loss_contrast

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        loss_contrast_meter.update(loss_contrast.item())
        pos_meter.update(pos_mask.sum(1).float().mean().item())
  
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_conf.update(mask.sum().item())
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.gpu==0:   
            lr_log = [pg['lr'] for pg in optimizer.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("{} || epoch:{}, iter: {}. loss_u: {:.3f}. loss_x: {:.3f}. loss_c: {:.3f}. "
                        "n_correct_u: {:.2f}/{:.2f}. n_edge: {:.3f}. LR: {:.2f}. "
                        "batch_time: {:.2f}. data_time: {:.2f}.".format(
                            args.exp_dir, epoch, i + 1, loss_u_meter.avg, loss_x_meter.avg, loss_contrast_meter.avg,
                            n_correct_u_lbs_meter.avg, n_conf.avg, pos_meter.avg, lr_log, batch_time.avg, data_time.avg))

    if args.gpu==0:    
        tb_logger.log_value('loss_x', loss_x_meter.avg, epoch)
        tb_logger.log_value('loss_u', loss_u_meter.avg, epoch)
        tb_logger.log_value('loss_c', loss_contrast_meter.avg, epoch)
        tb_logger.log_value('num_conf', n_conf.avg, epoch)
        tb_logger.log_value('guess_label_acc', n_correct_u_lbs_meter.avg/n_conf.avg, epoch)

        
        
def validate(val_loader, model, args, logger, tb_logger, epoch):
   
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            # compute output
            output,target = model(args, batch, is_eval=True)

            # measure accuracy 
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0])
            top5.update(acc5[0])

            if i % args.print_freq == 0 and args.gpu==0:
                logger.info("validation ||epoch:{}, iter: {}. acc1 : {:.2f}. acc5 : {:.2f}.".format(
                    epoch, i + 1, top1.avg, top5.avg))

    if args.gpu==0:    
        logger.info("validation ||epoch:{}, acc1 : {:.2f}. acc5 : {:.2f}.".format(epoch, top1.avg, top5.avg))
        tb_logger.log_value('test_acc', top1.avg, epoch)
        tb_logger.log_value('test_acc5', top5.avg, epoch)
    torch.cuda.empty_cache()    
    return top1.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """
    Computes and stores the average and current value

    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_default_logging(args, default_level=logging.INFO,
                          format="%(asctime)s - %(levelname)s -  %(message)s"):

    logger = logging.getLogger('')

    logging.basicConfig(  # unlike the root logger, a custom logger can’t be configured using basicConfig()
        filename=os.path.join(args.exp_dir, f'{time_str()}.log'),
        format=format,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=default_level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(default_level)
    console_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(console_handler)

    return logger

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'

    #     time.strftime(format[, t])
    return datetime.today().strftime(fmt)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
    
    
