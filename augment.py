import argparse
import math
import time

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, Dataset

from datasets import *
from models.network3 import Network3
from result_logger import ResultLogger
from utils import *

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
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
parser.add_argument('--new-size', type=int, default=256)
parser.add_argument('--crop-size', type=int, default=224)
parser.add_argument('--datadir', type=str, default='.')
parser.add_argument('--logdir', type=str, default='.')
parser.add_argument('--viz-step', type=int, default=1)
parser.add_argument('--warmup-threshold', type=float, default=None)
parser.add_argument('--warmup-epochs', type=int, default=0)
parser.add_argument('--lr-step', type=int, default=None)
parser.add_argument('--ngpus', default=1, type=int,
                    help='number of GPUs to use.')
parser.add_argument('--clip-grad', type=float, default=None)
parser.add_argument('--milestones', nargs='+', type=int, default=None)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--z-dim', type=int)
parser.add_argument('--h-dim', type=int)
parser.add_argument('--generative', action='store_true')

args = parser.parse_args()
free_gpus = get_free_gpu(num=args.ngpus)
os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus
# os.environ["OMP_NUM_THREADS"] = str(args.ngpus * 4)

best_acc1 = 0


class RepeatedDataset(Dataset):
    def __init__(self, mus, logvars, targets, rep_nums, class_cnts):
        super(RepeatedDataset, self).__init__()
        mus_list, logvars_list, targets_list = [], [], []
        n_classes = len(rep_nums)
        max_cnt = max(class_cnts)

        for i in range(n_classes):
            idx = (targets == i)
            mus_base = list(mus[idx])
            logvars_base = list(logvars[idx])
            targets_base = list(targets[idx])
            if rep_nums[i] <= 0.0:
                mus_list.extend(mus_base)
                logvars_list.extend(logvars_base)
                targets_list.extend(targets_base)
            else:
                mus_list.extend((mus_base * (math.ceil(rep_nums[i])))[:(max_cnt - class_cnts[i])])
                logvars_list.extend((logvars_base * (math.ceil(rep_nums[i])))[:(max_cnt - class_cnts[i])])
                targets_list.extend((targets_base * (math.ceil(rep_nums[i])))[:(max_cnt - class_cnts[i])])
        self.mus_list = mus_list
        self.logvars_list = logvars_list
        self.targets_list = targets_list

    def __getitem__(self, index):
        return self.mus_list[index], self.logvars_list[index], self.targets_list[index]

    def __len__(self):
        return len(self.mus_list)


def reparameterize(mu, logvar):
    std = (logvar.clamp(-50, 50).exp() + 1e-8) ** 0.5
    eps = torch.randn_like(logvar)
    return eps * std + mu


def get_data(loader, model, args):
    # switch to evaluate mode
    model.eval()

    mus = []
    logvars = []
    targets = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            targets.append(target)
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
            mu, logvar, _ = model.get_z(images)
            mus.append(mu.detach().cpu())
            logvars.append(logvar.detach().cpu())

        mus = torch.cat(mus, dim=0)
        logvars = torch.cat(logvars, dim=0)
        targets = torch.cat(targets, dim=0)
    return mus, logvars, targets


def get_rep_nums(dataset):
    class_cnts = dataset.get_class_count()
    max_cnt = max(class_cnts)
    rep_nums = [(max_cnt - class_cnts[i]) / class_cnts[i] for i in range(len(class_cnts))]
    print(rep_nums, class_cnts)
    return rep_nums, class_cnts


def augment_data(mus, logvars, target, rep_nums, class_cnts):
    mus_list, logvars_list, target_list = [], [], []
    n_classes = len(rep_nums)
    max_cnt = max(class_cnts)

    for i in range(n_classes):
        idx = (target == i)
        mus_base = mus[idx]
        logvars_base = logvars[idx]
        target_base = target[idx]
        if rep_nums[i] <= 0.0:
            mus_list.append(mus_base)
            logvars_list.append(logvars_base)
            target_list.append(target_base)
        else:
            mus_list.append(mus_base.repeat(math.ceil(rep_nums[i]), 1)[:(max_cnt - class_cnts[i])])
            logvars_list.append(logvars_base.repeat(math.ceil(rep_nums[i]), 1)[:(max_cnt - class_cnts[i])])
            target_list.append(target_base.repeat(math.ceil(rep_nums[i]))[:(max_cnt - class_cnts[i])])
    mus_list = torch.cat(mus_list, 0)
    logvars_list = torch.cat(logvars_list, 0)
    target_list = torch.cat(target_list, 0)
    return mus_list, logvars_list, target_list


def train_classifier(train_loader, model, criterion, optimizer, args, epoch, result_logger):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (mu, logvar, target) in enumerate(train_loader):
        if args.gpu is not None:
            mu = mu.cuda(args.gpu, non_blocking=True)
            logvar = logvar.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute loss
        z = reparameterize(mu, logvar)
        logits = model(z)
        loss = criterion(logits, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), target.size(0))
        top1.update(acc1[0].item(), target.size(0))
        top5.update(acc5[0].item(), target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)
        if torch.isnan(loss).any():
            raise RuntimeError("nan in loss!")


def validate(val_loader, model, criterion, args, epoch, result_logger=None):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    # switch to evaluate mode
    model.eval()

    probs = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (mu, logvar, target) in enumerate(val_loader):
            if args.gpu is not None:
                mu = mu.cuda(args.gpu, non_blocking=True)
                logvar = logvar.cuda(args.gpu, non_blocking=True)
            targets.append(target)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            z = reparameterize(mu, logvar)
            logits = model(z)
            prob = torch.softmax(logits, dim=1)

            probs.append(prob.detach().cpu())
            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            top1.update(acc1[0].item(), target.size(0))
            top5.update(acc5[0].item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        probs = torch.cat(probs, dim=0)
        predicts = torch.argmax(probs, dim=1)
        targets = torch.cat(targets, dim=0)
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if result_logger is not None:
        result_logger.add_test_metrics(targets.numpy(), predicts.numpy(), probs.numpy(), time=batch_time.sum)
    return top1.avg


def save_checkpoint(state, is_best, best_filename='checkpoint.pth', logdir=None):
    path = os.path.join(logdir if logdir is not None else os.getcwd(), best_filename)
    torch.save(state, path)
    if is_best:
        best_filename = 'fc_best_ep{:03d}_acc{:.2f}.pth'.format(state['epoch'], state['best_acc1'])
        shutil.copyfile(path, os.path.join(logdir, best_filename))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
new_size, crop_size = args.new_size, args.crop_size
batch_size = args.batch_size
transform = transforms.Compose([
    transforms.Resize((new_size, new_size)),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    normalize,
])
train_dataset = get_dataset(args.data, root=args.datadir, train=True, transform=transform)
val_dataset = get_dataset(args.data, root=args.datadir, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)
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

if os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume, map_location='cpu')
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}  # remove "module." prefix in keys

    model.load_state_dict(state_dict)
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))

# load weight
torch.cuda.set_device(args.gpu)
model = model.cuda(args.gpu)
train_mus, train_logvars, train_target = get_data(train_loader, model, args)
test_mus, test_logvars, test_target = get_data(val_loader, model, args)

model = model.classifier
rep_nums, class_cnts = get_rep_nums(train_dataset)
# train_mus, train_logvars, train_target = augment_data(train_mus, train_logvars, train_target, rep_nums, class_cnts)

train_loader = torch.utils.data.DataLoader(
    RepeatedDataset(train_mus, train_logvars, train_target, rep_nums, class_cnts),
    batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    TensorDataset(test_mus, test_logvars, test_target), batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

train_class_count = train_dataset.get_class_count()
test_class_count = val_dataset.get_class_count()
result_logger = ResultLogger('metrics_aug', args.num_classes, train_class_count, test_class_count,
                             args.logdir, verbose=False)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)

for epoch in range(args.start_epoch, args.epochs):
    train_classifier(train_loader, model, criterion, optimizer, args, epoch, result_logger)
    acc1 = validate(val_loader, model, criterion, args, epoch, result_logger)

    scheduler.step()
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    result_logger.save_metrics()
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer': optimizer.state_dict(),
    }, is_best, logdir=args.logdir)

print(best_acc1)
