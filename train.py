import argparse
import json
import random
import time
import warnings
import math
from pathlib import Path
from shutil import copyfile

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from models.GMMLoss import GMMLoss
from models.network3 import Network3
from result_logger import ResultLogger
from utils import *

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('data', metavar='DIR',
                    help='path to datasets')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--step-size', default=30, type=int)
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--new-size', type=int, default=256,
                    help='size of the resized input image')
parser.add_argument('--crop-size', type=int, default=224,
                    help='size of the croped input image')
parser.add_argument('--datadir', type=str, default='.',
                    help='path to the dataset')
parser.add_argument('--logdir', type=str, default='.',
                    help='path to the log directory')
parser.add_argument('--warmup-threshold', type=float, default=None,
                    help='warm up until the loss less than given threshold')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='warm up epochs')
parser.add_argument('--lr-step', type=int, default=None,
                    help='parameter for StepLR scheduler')
parser.add_argument('--ngpus', default=1, type=int,
                    help='number of GPUs to use.')
parser.add_argument('--milestones', nargs='+', type=int, default=None,
                    help='parameter for MultiStepLR scheduler')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='parameter for scheduler')
parser.add_argument('--clip-grad', type=float, default=None,
                    help='clip gradient norm')
parser.add_argument('--z-dim', type=int,
                    help='dimension for z')
parser.add_argument('--h-dim', type=int,
                    help='dimension for latent space')
parser.add_argument('--generative', action='store_true',
                    help='whether to use generative classifier')
parser.add_argument('--weight-cls', type=float, default=1.0,
                    help='loss weight for classification term')
parser.add_argument('--weight-reconst', type=float, default=1.0,
                    help='loss weight for reconstruction term')
parser.add_argument('--weight-gmm', type=float, default=1.0,
                    help='loss weight for GMM term')
best_acc1 = 0
best_gma = 0


def main():
    args = parser.parse_args()
    free_gpus = get_free_gpu(num=args.ngpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus
    os.environ["OMP_NUM_THREADS"] = str(args.ngpus * 4)

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    with open(os.path.join(args.logdir, 'args.txt'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    filepath = Path(__file__)
    copyfile(filepath.absolute(), os.path.join(args.logdir, filepath.name))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_gma
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    num_classes = args.num_classes = get_dataset_class_number(args.data)
    if args.arch == 'resnet_vae3':
        from models.ResNetVAE3 import ResNet_VAE_encoder
        from models.ResNetVAE3 import ResNet_VAE_decoder
        encoder = ResNet_VAE_encoder(h_dim=args.h_dim, pretrained=args.pretrained)
        decoder = ResNet_VAE_decoder(num_classes, h_dim=args.h_dim, z_dim=args.z_dim, img_size=args.crop_size)
        net = Network3
    else:
        raise NotImplementedError
    model = net(encoder, decoder, num_classes, args.h_dim, args.z_dim, generative=args.generative, mlp_dim=None)
    criterion = {'reconst': nn.MSELoss(),
                 'cls': nn.CrossEntropyLoss(),
                 'GMM': GMMLoss(num_classes, args.z_dim)}
    eval_criterion = {'reconst': nn.MSELoss(reduction='none'),
                      'cls': nn.CrossEntropyLoss(reduction='none'),
                      'GMM': GMMLoss(num_classes, args.z_dim, reduction='none')}

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            criterion['GMM'].cuda(args.gpu)
            eval_criterion['GMM'].cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            criterion['GMM'].cuda()
            eval_criterion['GMM'].cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        criterion['GMM'].cuda(args.gpu)
        eval_criterion['GMM'].cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
        criterion['GMM'].cuda()
        eval_criterion['GMM'].cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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
            best_acc1 = torch.tensor(checkpoint['best_acc1'])
            best_gma = torch.tensor(checkpoint['best_gma'])
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
                best_gma = best_gma.to(args.gpu)
            if not args.distributed:
                checkpoint['state_dict'] = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    new_size, crop_size = args.new_size, args.crop_size

    train_dataset = get_dataset(args.data, root=args.datadir, train=True, transform=transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    val_dataset = get_dataset(args.data, root=args.datadir, train=False, transform=transforms.Compose([
        transforms.Resize((new_size, new_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    args.is_master = not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    if args.warmup_epochs > 0:
        print('=> pre-training')
        for epoch in range(args.warmup_epochs):
            warmup(train_loader, model, criterion, optimizer, args, args.warmup_threshold)
        print('=> finish pre-training')
    set_parameter_requires_grad(model, True)

    if args.lr_step is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
    elif args.milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.T_max is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=1e-5)
    else:
        raise ValueError('Unknown LR scheduler')
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    writer = SummaryWriter(args.logdir) if args.is_master else None
    train_class_count = train_dataset.get_class_count()
    test_class_count = val_dataset.get_class_count()
    result_logger = ResultLogger('metrics', args.num_classes, train_class_count, test_class_count,
                                 args.logdir, verbose=False) if args.is_master else None
    if args.evaluate:
        validate(val_loader, model, eval_criterion, args, 0, None, result_logger)
        if result_logger is not None:
            result_logger.save_metrics()
        return
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, args, epoch, writer, result_logger)

        # evaluate on validation set
        acc1, g_macro = validate(val_loader, model, eval_criterion, args, epoch, writer, result_logger)

        # if epoch % 2 == 0:
        #     scheduler.step(epoch // 2)
        scheduler.step()
        if writer is not None:
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
        if result_logger is not None:
            result_logger.save_metrics()
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1 or g_macro > best_gma

        best_acc1 = max(acc1, best_acc1)
        best_gma = max(g_macro, best_gma)

        if args.is_master:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_gma': best_gma,
                'optimizer': optimizer.state_dict(),
            }, is_best, logdir=args.logdir)
            writer.flush()


def train(train_loader, model, criterion, optimizer, args, epoch, writer, result_logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    ce_losses = AverageMeter('CELoss', ':.4e')
    reconst_losses = AverageMeter('ReconLoss', ':.4e')
    gmm_losses = AverageMeter('GMMLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, ce_losses, reconst_losses, gmm_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    for cri in criterion.values():
        cri.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute loss
        target_onehot = F.one_hot(target, num_classes=args.num_classes).float()  # if args.generative else None
        mu, logvar, z, logits, x_reconst = model(images, target_onehot)
        cls_loss = criterion['cls'](logits, target) * args.weight_cls
        reconst_loss = criterion['reconst'](x_reconst, images) * args.weight_reconst
        gmm_loss = criterion['GMM'](mu, logvar, z) * args.weight_gmm
        loss = reconst_loss + cls_loss + gmm_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        ce_losses.update(cls_loss.item(), images.size(0))
        reconst_losses.update(reconst_loss.item(), images.size(0))
        gmm_losses.update(gmm_loss.item(), images.size(0))

        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")

    if writer is not None:
        writer.add_scalar('Time/train', batch_time.avg, epoch)
        writer.add_scalars('Losses/train', {'total_loss': losses.avg,
                                            'ce_loss': ce_losses.avg,
                                            'reconst_loss': reconst_losses.avg,
                                            'gmm_loss': gmm_losses.avg
                                            }, epoch)
        writer.add_scalar('Accuracy1/train', top1.avg, epoch)
        writer.add_scalar('Accuracy5/train', top5.avg, epoch)
    if result_logger is not None:
        result_logger.add_training_metrics(losses.avg, top1.avg, batch_time.sum)


def validate(val_loader, model, criterion, args, epoch=None, writer=None, result_logger=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    for cri in criterion.values():
        cri.eval()

    probs = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            targets.append(target)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.generative:
                if args.distributed:
                    # class_per_gpu = math.ceil(args.num_classes / args.ngpus)
                    # from_class = class_per_gpu * args.gpu
                    # to_class = min(class_per_gpu * (args.gpu + 1), args.num_classes)
                    class_per_gpu = round(args.num_classes / args.ngpus)
                    from_class = class_per_gpu * args.gpu
                    to_class = class_per_gpu * (args.gpu + 1) if args.gpu != args.ngpus - 1 else args.num_classes
                else:
                    from_class = 0
                    to_class = args.num_classes
                elbos = torch.zeros(images.size(0), class_per_gpu if args.distributed else args.num_classes,
                                    device=images.device)

                for j, c in enumerate(range(from_class, to_class)):
                    target_onehot = torch.zeros(images.size(0), args.num_classes, device=images.device)
                    target_onehot[:, c] = 1.0
                    mu, logvar, z, logits, x_reconst = model(images, target_onehot)
                    cls_loss = criterion['cls'](logits, torch.full_like(target, c)) * args.weight_cls
                    reconst_loss = criterion['reconst'](x_reconst, images).mean([1, 2, 3]) * args.weight_reconst
                    gmm_loss = criterion['GMM'](mu, logvar, z) * args.weight_gmm
                    elbo = reconst_loss + cls_loss + gmm_loss
                    elbos[:, j] = elbo

                if args.distributed:
                    elbo_list = [torch.empty_like(elbos) for _ in range(args.ngpus)]
                    dist.all_gather(elbo_list, elbos)
                    elbos = torch.cat(elbo_list, dim=1)
                    elbos = elbos[:, :args.num_classes]
                logits = - elbos
            else:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    logits = model.module.get_logits(images, None)
                else:
                    logits = model.get_logits(images, None)
            prob = torch.softmax(logits, dim=1)

            probs.append(prob.detach().cpu())
            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.is_master and i % args.print_freq == 0:
                progress.display(i)

        probs = torch.cat(probs, dim=0)
        predicts = torch.argmax(probs, dim=1)
        targets = torch.cat(targets, dim=0)
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if writer is not None:
        writer.add_scalar('Time/test', batch_time.avg, epoch)
        writer.add_scalar('Accuracy1/test', top1.avg, epoch)
        writer.add_scalar('Accuracy5/test', top5.avg, epoch)

    g_macro = 0.0
    if result_logger is not None:
        g_macro = result_logger.add_test_metrics(targets.numpy(), predicts.numpy(), probs.numpy(), time=batch_time.sum)
    return top1.avg, g_macro


def warmup(train_loader, model, criterion, optimizer, args, warmup_threshold=None):
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [losses],
        prefix="Warmup: ")

    # switch to train mode
    model.train()
    for cri in criterion.values():
        cri.train()

    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute loss
        target_onehot = F.one_hot(target, num_classes=args.num_classes).float() if args.generative else None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            logits = model.module.get_logits(images, target_onehot)
        else:
            logits = model.get_logits(images, target_onehot)

        loss = F.cross_entropy(logits, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        if i % args.print_freq == 0:
            progress.display(i)
        if warmup_threshold is not None and i * args.batch_size > 256 and losses.avg < warmup_threshold:
            print("=> Avg loss lower than threshold, stop warming up")
            break
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")


# def warmup(train_loader, model, criterion, optimizer, args, warmup_threshold=0.1):
#     losses = AverageMeter('Loss', ':.4e')
#     progress = ProgressMeter(
#         len(train_loader),
#         [losses],
#         prefix="Warmup: ")
#
#     # switch to train mode
#     model.train()
#     for cri in criterion.values():
#         cri.train()
#
#     for i, (images, target) in enumerate(train_loader):
#         if args.gpu is not None:
#             images = images.cuda(args.gpu, non_blocking=True)
#         if torch.cuda.is_available():
#             target = target.cuda(args.gpu, non_blocking=True)
#
#         # compute loss
#         target_onehot = F.one_hot(target, num_classes=args.num_classes).float() if args.generative else None
#         if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#             mu, logvar, _ = model.module.get_z(images, target_onehot)
#         else:
#             mu, logvar, _ = model.get_z(images, target_onehot)
#
#         mean_loss = F.mse_loss(mu, torch.zeros_like(mu), reduction='mean')
#         cov_loss = F.mse_loss(logvar, torch.zeros_like(logvar), reduction='mean')
#         loss = mean_loss + cov_loss
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         if args.clip_grad:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
#         optimizer.step()
#
#         if i % args.print_freq == 0:
#             progress.display(i)
#         if i * args.batch_size > 256 and losses.avg < warmup_threshold:
#             print("=> Avg loss lower than threshold, stop warming up")
#             break
#         if torch.isnan(loss).any():
#             raise RuntimeError("nan in loss!")


if __name__ == '__main__':
    main()
