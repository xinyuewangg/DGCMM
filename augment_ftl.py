import argparse
import math
import time

import h5py
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset

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
parser.add_argument('--weight-cls', type=float, default=1.0)
parser.add_argument('--weight-reconst', type=float, default=1.0)
parser.add_argument('--weight-gmm', type=float, default=1.0)
parser.add_argument('--rep-method', type=str, default='get_rep_nums1')

args = parser.parse_args()
free_gpus = get_free_gpu(num=args.ngpus)
os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus
os.environ["OMP_NUM_THREADS"] = str(args.ngpus * 4)

best_acc1 = 0


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
            # if i > 10:
            #     break

        mus = torch.cat(mus, dim=0)
        logvars = torch.cat(logvars, dim=0)
        targets = torch.cat(targets, dim=0)
    return mus, logvars, targets


def get_rep_nums1(dataset):
    class_cnts = dataset.get_class_count()
    max_cnt = max(class_cnts)
    rep_nums = [(max_cnt - class_cnts[i]) / class_cnts[i] for i in range(len(class_cnts))]
    return rep_nums, class_cnts


def get_rep_nums2(dataset, many_shot_thr=100, low_shot_thr=20):
    class_cnts = dataset.get_class_count()
    max_cnt = max(class_cnts)
    min_many_cnt = min(cnt for cnt in class_cnts if cnt > many_shot_thr)
    min_median_cnt = min(cnt for cnt in class_cnts if low_shot_thr <= cnt <= many_shot_thr)
    rep_nums = []
    for cnt in class_cnts:
        if cnt > many_shot_thr:
            num = (max_cnt - cnt) / cnt
        elif cnt < low_shot_thr:
            num = (min_median_cnt - cnt) / cnt
        else:
            num = (min_many_cnt - cnt) / cnt
        rep_nums.append(num)
    return rep_nums, class_cnts


def update_stats(mus, target, num_classes):
    zdim = mus.size(1)
    covar = torch.zeros(zdim, zdim).to(mus.device)
    centers = []
    for c in range(num_classes):
        mus_c = mus[target == c]
        cc = mus_c.mean(0, True)
        centers.append(cc.squeeze())
        mus_cc = mus_c - cc
        var = mus_cc.t() @ mus_cc
        covar += var
    U, S, V = torch.pca_lowrank(covar)
    Q = V[:, :zdim // 2]
    QQ = Q @ Q.t()
    return QQ, torch.stack(centers, dim=0)


def split_class(class_cnts, many_shot_thr=100):
    many_cls = []
    few_cls = []
    for i, cnt in enumerate(class_cnts):
        if cnt > many_shot_thr:
            many_cls.append((cnt, i))
        else:
            few_cls.append((cnt, i))
    return [x for _, x in sorted(many_cls, reverse=True, key=lambda x: x[0])], \
           [x for _, x in sorted(few_cls, reverse=True, key=lambda x: x[0])]


def get_class_data(target, cls, cat=False):
    indices = []
    for c in cls:
        idx = (target == c).nonzero(as_tuple=False).squeeze()
        indices.append(idx)
    if cat:
        indices = torch.cat(indices, dim=0)
    return indices


def augment_batch(mus, logvars, target: torch.Tensor):
    z_final, y_final = [], []

    for i in range(num_classes):
        idx = target == i
        num_i = idx.count_nonzero()
        if num_i == 0:
            continue

        raw_mus_i = mus[idx]
        raw_logvars_i = logvars[idx]
        z_raw_i = reparameterize(raw_mus_i, raw_logvars_i)
        z_final.append(z_raw_i)

        aug_num = int(num_i * rep_nums[i])
        if aug_num > 0:
            rand_idx = torch.randint(0, len(augment_dict[i]), [aug_num])
            mus_j, logvars_j, target_j = train_mus[augment_dict[i][rand_idx]], \
                                         train_logvars[augment_dict[i][rand_idx]], \
                                         train_target[augment_dict[i][rand_idx]]
            z_j = reparameterize(mus_j, logvars_j)
            z_aug_i = centers[i] + (z_j - centers[target_j]) @ covar
            z_final.append(z_aug_i)
        else:
            z_aug_i = []
        y_final.append(torch.full([len(z_raw_i) + len(z_aug_i)], i))
    return torch.cat(z_final, 0), torch.cat(y_final, 0)


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
        z, target = augment_batch(mu, logvar, target)
        if args.gpu is not None:
            z = z.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute loss
        # z = reparameterize(mu, logvar)
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
        best_filename = 'fc_ftl_best_ep{:03d}_acc{:.2f}.pth'.format(state['epoch'], state['best_acc1'])
        shutil.copyfile(path, os.path.join(logdir, best_filename))


def save_features(data, save_path):
    train_mus, train_logvars, train_target, test_mus, test_logvars, test_target = data
    with h5py.File(save_path, 'w') as f:
        f.create_dataset("train_mus", data=train_mus)
        f.create_dataset("train_logvars", data=train_logvars)
        f.create_dataset("train_target", data=train_target)
        f.create_dataset("test_mus", data=test_mus)
        f.create_dataset("test_logvars", data=test_logvars)
        f.create_dataset("test_target", data=test_target)


def load_features(save_path):
    with h5py.File(save_path, 'r') as f:
        train_mus = f['train_mus'].value
        train_logvars = f['train_logvars'].value
        train_target = f['train_target'].value
        test_mus = f['test_mus'].value
        test_logvars = f['test_logvars'].value
        test_target = f['test_target'].value
    return train_mus, train_logvars, train_target, test_mus, test_logvars, test_target


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
save_path = os.path.join(args.logdir, "all_features.h5")
if os.path.isfile(save_path):
    train_mus, train_logvars, train_target, test_mus, test_logvars, test_target = load_features(save_path)
else:
    train_mus, train_logvars, train_target = get_data(train_loader, model, args)
    test_mus, test_logvars, test_target = get_data(val_loader, model, args)
    save_features([train_mus, train_logvars, train_target, test_mus, test_logvars, test_target], save_path)

model = model.classifier
# model = nn.Sequential(nn.Linear(2048, 1024),
#                       nn.ReLU(True),
#                       nn.Linear(1024, num_classes))
if args.rep_method == 'get_rep_nums1':
    rep_nums, class_cnts = get_rep_nums1(train_dataset)
elif args.rep_method == 'get_rep_nums2':
    rep_nums, class_cnts = get_rep_nums2(train_dataset)
else:
    raise NotImplementedError
covar, centers = update_stats(train_mus, train_target, len(rep_nums))
# train_mus, train_logvars, train_target = augment_data(train_mus, train_logvars, train_target, rep_nums, class_cnts)
many_cls, few_cls = split_class(class_cnts)
augment_dict = {}
for c in tqdm(range(num_classes)):
    if c in many_cls:
        idx = many_cls.index(c)
        aug_cls_list = many_cls[:idx]
    else:
        aug_cls_list = many_cls
    if len(aug_cls_list) > 0:
        aug_source = get_class_data(train_target, aug_cls_list, cat=True)
        augment_dict[c] = aug_source
    # else:
    #     print(c, class_cnts[c])

train_loader = torch.utils.data.DataLoader(
    TensorDataset(train_mus, train_logvars, train_target), batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(
    TensorDataset(test_mus, test_logvars, test_target), batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), args.lr,
#                              weight_decay=args.weight_decay)

train_class_count = train_dataset.get_class_count()
test_class_count = val_dataset.get_class_count()
result_logger = ResultLogger('metrics_aug_ftl', args.num_classes, train_class_count, test_class_count,
                             args.logdir, verbose=False)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.gamma)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

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
