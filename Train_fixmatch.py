from __future__ import print_function
import random

import time
import argparse
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from WideResNet import WideResnet
from datasets.cifar import get_train_loader, get_val_loader

from utils import accuracy, setup_default_logging, AverageMeter, WarmupCosineLrScheduler

import tensorboard_logger 

def set_model(args):
    model = WideResnet(n_classes=args.n_classes,k=args.wresnet_k, n=args.wresnet_n, proj=False)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)

        msg = model.load_state_dict(checkpoint, strict=False)
        assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
        assert set(msg.unexpected_keys) == {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}
        print('loaded from checkpoint: %s'%args.checkpoint)
            
    model.train()
    model.cuda()
    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()
    
    if args.eval_ema:
        ema_model = WideResnet(n_classes=args.n_classes,k=args.wresnet_k, n=args.wresnet_n, proj=False)
        for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
        ema_model.cuda()  
        ema_model.eval()
    else:
        ema_model = None    
              
    return model, criteria_x, criteria_u, ema_model


@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1-ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)    
        
        
def train_one_epoch(epoch,
                    model,
                    ema_model,
                    criteria_x,
                    criteria_u,
                    optim,
                    lr_schdlr,
                    dltrain_x,
                    dltrain_u,
                    args,
                    n_iters,
                    logger,
                    prob_list,
                    ):
    model.train()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    # the number of correctly-predicted and gradient-considered unlabeled data
    n_correct_u_lbs_meter = AverageMeter()
    # the number of gradient-considered strong augmentation (logits above threshold) of unlabeled samples
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()

    
    epoch_start = time.time()  # start time
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters):
        ims_x_weak, lbs_x = next(dl_x)
        (ims_u_weak, ims_u_strong), lbs_u_real = next(dl_u)

        lbs_x = lbs_x.cuda()
        lbs_u_real = lbs_u_real.cuda()

        # --------------------------------------
        bt = ims_x_weak.size(0)
        mu = int(ims_u_weak.size(0) // bt)
        imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong], dim=0).cuda()
        logits = model(imgs)

        logits_x = logits[:bt]
        logits_u_w, logits_u_s = torch.split(logits[bt:], bt * mu)

        loss_x = criteria_x(logits_x, lbs_x)

        with torch.no_grad():
            probs = torch.softmax(logits_u_w, dim=1)
            
            if args.DA:
                prob_list.append(probs.mean(0))
                if len(prob_list)>32:
                    prob_list.pop(0)
                prob_avg = torch.stack(prob_list,dim=0).mean(0)
                probs = probs / prob_avg
                probs = probs / probs.sum(dim=1, keepdim=True)                  
            
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args.thr).float()

            probs = probs.detach()

        loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean()

        loss = loss_x + args.lam_u * loss_u 
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()
        
        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)        

        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        mask_meter.update(mask.mean().item())

        
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())

        if (it + 1) % 64 == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}. loss_u: {:.3f}. loss_x: {:.3f}. "
                        "n_correct_u: {:.2f}/{:.2f}. Mask:{:.3f}. LR: {:.3f}. Time: {:.2f}".format(
                args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_u_meter.avg, loss_x_meter.avg, 
                n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, lr_log, t))

            epoch_start = time.time()

    return loss_x_meter.avg, loss_u_meter.avg, mask_meter.avg, n_correct_u_lbs_meter.avg/n_strong_aug_meter.avg, prob_list


def evaluate(model, ema_model, dataloader, criterion):
    
    model.eval()

    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()

    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            
            logits = model(ims)
            loss = criterion(logits, lbs)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())
            
            if ema_model is not None:
                logits = ema_model(ims)
                loss = criterion(logits, lbs)
                scores = torch.softmax(logits, dim=1)
                top1, top5 = accuracy(scores, lbs, (1, 5))                
                ema_top1_meter.update(top1.item())

    return top1_meter.avg, ema_top1_meter.avg


def main():
    parser = argparse.ArgumentParser(description='FixMatch Training')
    parser.add_argument('--root', default='./data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')    
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=10,
                         help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=40,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,
                        help='factor of train batch size of unlabeled samples')
    
    parser.add_argument('--eval-ema', default=True, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)    

    parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024,
                        help='number of training images for each epoch')
    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random behaviors, no seed if negtive')
    parser.add_argument('--DA', default=True, help='use distribution alignment')

    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')   
    
    parser.add_argument('--exp-dir', default='FixMatch', type=str, help='experiment directory')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')
    #/export/home/project/SimCLR/save_cifar_t0.2/checkpoint_100.tar
    
    args = parser.parse_args()
    
    logger, output_dir = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))
    
    tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)

    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize  # 1024
    n_iters_all = n_iters_per_epoch * args.n_epoches  # 1024 * 200

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")
    
    model, criteria_x, criteria_u, ema_model = set_model(args)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    dltrain_x, dltrain_u = get_train_loader(
        args.dataset, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled, root=args.root, method='fixmatch')
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2)

    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)  
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum, nesterov=True)

    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)
    
    prob_list = []
    train_args = dict(
        model=model,
        ema_model=ema_model,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger,
        prob_list=prob_list
    )
    best_acc = -1
    best_epoch = 0
    logger.info('-----------start training--------------')
    for epoch in range(args.n_epoches):
        loss_x, loss_u, mask_mean, guess_label_acc, prob_list = train_one_epoch(epoch, **train_args)

        top1, ema_top1 = evaluate(model, ema_model, dlval, criteria_x)
    
        tb_logger.log_value('loss_x', loss_x, epoch)
        tb_logger.log_value('loss_u', loss_u, epoch)
        tb_logger.log_value('guess_label_acc', guess_label_acc, epoch)
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('test_ema_acc', ema_top1, epoch)
        tb_logger.log_value('mask', mask_mean, epoch)
        
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch

        logger.info("Epoch {}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, ema_top1, best_acc, best_epoch))


if __name__ == '__main__':
    main()
