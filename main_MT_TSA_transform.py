from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import argparse

import numpy as np
import shutil
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms

import torch.nn.functional as F

from ImageDataLoader_MT import SimpleImageLoader
from models import Res18, Res50, Dense121, Res18_basic

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

NUM_CLASSES = 265

#####  BASE LINE FUNFCTIONS FOR TRAINING AND NSML USAGE START
#####  -->
def top_n_accuracy_score(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx + 1:]:
            counter += 1
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


def adjust_learning_rate(opts, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opts.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, output_x1, output_x2, targets_x, output_u1, output_u2, epoch, final_epoch):
        # compute classification loss for train data
        x1_loss = torch.sum(F.log_softmax(output_x1, dim=1) * targets_x, dim=1)
        Lclass = -torch.mean(x1_loss)
        # compute consistency loss for unlabeled data
        x_se = (output_x1 - output_x2) ** 2
        u_se = (output_u1 - output_u2) ** 2
        cat_se = torch.cat((x_se, u_se), 0)
        Lcon = torch.mean(cat_se)
        return Lclass, Lcon, linear_rampup(epoch, opts.rampup_length)


class WeightEMA(object):
    def __init__(self, model, ema_model):
        self.model = model
        self.ema_model = ema_model
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self, alpha):
        one_minus_alpha = 1.0 - alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(alpha)
                ema_param.add_(param * one_minus_alpha)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def split_ids(path, ratio):
    with open(path) as f:
        ids_l = []
        ids_u = []
        for i, line in enumerate(f.readlines()):
            if i == 0 or line == '' or line == '\n':
                continue
            line = line.replace('\n', '').split('\t')
            if int(line[1]) >= 0:
                ids_l.append(int(line[0]))
            else:
                ids_u.append(int(line[0]))

    ids_l = np.array(ids_l)
    ids_u = np.array(ids_u)

    perm = np.random.permutation(np.arange(len(ids_l)))
    cut = int(ratio * len(ids_l))
    train_ids = ids_l[perm][cut:]
    val_ids = ids_l[perm][:cut]

    return train_ids, val_ids, ids_u


### NSML functions
def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(root_path, 'test',
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.CenterCrop(opts.imsize),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ])), batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True)
        print('loaded {} test images'.format(len(test_loader.dataset)))

    outputs = []
    s_t = time.time()
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        _, probs = model(image)
        output = torch.argmax(probs, dim=1)
        output = output.detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs)
    return outputs


def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = model.state_dict()
        torch.save(state, os.path.join(dir_name, 'model.pt'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)
######  <-- BASE LINE FUNCTIONS END #######

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Sample Product200K Training')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--steps_per_epoch', type=int, default=30, metavar='N',
                    help='number of steps to train per epoch (-1: num_data//batchsize)')

# basic settings
parser.add_argument('--name', default='Res18baseMM', type=str, help='output model name')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--seed', type=int, default=123, help='random seed')

# hyper-parameters for time signal annealing
# there are 3 options A/B/C and only one of them is activated for each Test
# to test which option works best, we activated one of the three for each tests
#setting A
parser.add_argument('--no_trainaug', default=2, type=int, help='number of augmentation for each training data')
parser.add_argument('--batchsize', default=200, type=int, help='traing data batchsize')
parser.add_argument('--unlbatchsize', default=200, type=int, help='unlabeled data batchsize')
parser.add_argument('--valbatchsize', default=265, type=int, help='validation data batchsize')
parser.add_argument('--epochdrop', default=100, type=int, help='epoch of which batch size drops')
parser.add_argument('--tbs_d', default=100, type=int, help='batchsized for epoch drop')
parser.add_argument('--utb_d', default=400, type=int, help='batchsize_unl for epoch drop')
parser.add_argument('--epochdropdrop', default=150, type=int, help='epoch of which batch size drops again')
parser.add_argument('--tbs_dd', default=50, type=int, help='batchsize for epoch double drop')
parser.add_argument('--utb_dd', default=500, type=int, help='batchsize_unl for epoch double drop')
'''
#setting B
parser.add_argument('--no_trainaug', default=4, type=int, help='number of augmentation for each training data')
parser.add_argument('--batchsize', default=100, type=int, help='traing data batchsize')
parser.add_argument('--unlbatchsize', default=200, type=int, help='unlabeled data batchsize')
parser.add_argument('--valbatchsize', default=265, type=int, help='validation data batchsize')
parser.add_argument('--epochdrop', default=100, type=int, help='epoch of which batch size drops')
parser.add_argument('--tbs_d', default=50, type=int, help='batchsized for epoch drop')
parser.add_argument('--utb_d', default=400  , type=int, help='batchsize_unl for epoch drop')
parser.add_argument('--epochdropdrop', default=150, type=int, help='epoch of which batch size drops again')
parser.add_argument('--tbs_dd', default=25, type=int, help='batchsize for epoch double drop')
parser.add_argument('--utb_dd', default=500, type=int, help='batchsize_unl for epoch double drop')
'''
'''
#setting C
parser.add_argument('--no_trainaug', default=4, type=int, help='number of augmentation for each training data')
parser.add_argument('--batchsize', default=100, type=int, help='traing data batchsize')
parser.add_argument('--unlbatchsize', default=200, type=int, help='unlabeled data batchsize')
parser.add_argument('--valbatchsize', default=265, type=int, help='validation data batchsize')
parser.add_argument('--epochdrop', default=100, type=int, help='epoch of which batch size drops')
parser.add_argument('--tbs_d', default=50, type=int, help='batchsized for epoch drop')
parser.add_argument('--utb_d', default=400  , type=int, help='batchsize_unl for epoch drop')
parser.add_argument('--epochdropdrop', default=150, type=int, help='epoch of which batch size drops again')
parser.add_argument('--tbs_dd', default=10, type=int, help='batchsize for epoch double drop')
parser.add_argument('--utb_dd', default=560, type=int, help='batchsize_unl for epoch double drop')

'''
# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')
parser.add_argument('--ema_decay', type=float, default=0.999, help='ema decay rate (0: no ema model)')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=50, help='saving epoch interval')

# hyper-parameters for mean teacher
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=3, type=float)
parser.add_argument('--rampup_length', default=80, type=float)

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml 
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

################################
################################

# These options are for parametrization of transform augmenatation types to be used.
# However, if randomize is selected True, rest of the options will be ignored,
# and the two will be randomly selected: 1) number of tranforms 2) Transform types
# If randomize is False and n is not 0, n number of transform types will be randomly selected
# if randomize is False and n is 0, customized transform types will be selected to be used.
parser.add_argument('--randomize', type=bool, default=False, help='what types?')
parser.add_argument('--n', type=int, default=3, help='0~4. if 0: follow the rest settings')
parser.add_argument('--resize_crop', type=bool, default=True, help='what types?')
parser.add_argument('--gray', type=bool, default=False, help='what types?')
parser.add_argument('--horizontal', type=bool, default=True, help='what types?')
parser.add_argument('--jitter', type=bool, default=False, help='what types?')
parser.add_argument('--rotate', type=bool, default=True, help='what types?')
parser.add_argument('--vertical', type=bool, default=False, help='what types?')


# function for returning transform = [transforms.example_1(), transforms.example_n()]
# prior to previous baseline code which specified which transform types to be used during the main(),
# utilizing this function enables the options chosen above to be selected as descibeda above.
def TransformType(imResize, imsize, n=0, randomize=False, affine=False, rotate=False, resize_crop=True, gaussian=False,
                  gray=False,
                  horizontal=False,
                  vertical=False, jitter=False):
    combined = [transforms.Resize(imResize)]
    if randomize:
        # n is selected randomly from 1, 2, 3, 4
        # (only up to 4, because too much transforms could rather hinder training)
        n = int(random.uniform(1, 4))
    if n:
        # all the customizations which could've been possible selected are all turned False.
        # only the ones selected from random.sample will be turned True.
        randomize, affine, rotate, gaussian, gray, horizontal, vertical, jitter = False, False, False, False, False, False, False, False
        randomly_selected = random.sample(['gray', 'rotate', 'horizontal', 'jitter'], n)
        if 'rotate' in randomly_selected:
            rotate = True
        if 'gray' in randomly_selected:
            gray = True
        if 'horizontal' in randomly_selected:
            horizontal = True
        if 'jitter' in randomly_selected:
            jitter = True
    # All the parameters' default values are as it was in the baseline code.
    # if n is not 0 and randomize is False, Other options will be considered in customizing
    # transform types

    if resize_crop:
        combined += [transforms.RandomResizedCrop(imsize)]
    if rotate:
        # Too much rotation will not give a good transform result compared to
        # real validity test images.
        combined += [transforms.RandomRotation(degrees=(-35, 35))]
    if gray:
        combined += [transforms.RandomGrayscale()]
    if affine:
        x, y = (random.uniform(0, 0.4), random.uniform(0, 0.4))
        combined += [transforms.RandomAffine(0, translate=(x, y))]
    if horizontal:
        combined += [transforms.RandomHorizontalFlip()]
    if vertical:
        # vertical is chosen not to be selected during randomization setting.
        # vertical transform seems not suitable for the shopping images since none of them are
        # intentionally inverted upside down.
        combined += [transforms.RandomVerticalFlip()]
    if jitter:
        # again, not too much degree of transform will be favorable, so the values are limited to
        # 0~0.3 for hue, and 0~2 for the rest of jitter function.
        rand_bright_num, rand_cont_num, rand_satur_num, rand_hue_num = random.uniform(0, 2), random.uniform(0, 2), \
                                                                       random.uniform(0, 2), random.uniform(0, 0.3),
        combined += [
            transforms.ColorJitter(brightness=rand_bright_num, contrast=rand_cont_num, saturation=rand_satur_num,
                                   hue=rand_hue_num)]
        """
        rand_bright_switch, rand_cont_switch, rand_satur_switch, rand_hue_switch = bool(random.getrandbits(1)), 
        bool(random.getrandbits(1)), bool(random.getrandbits(1)), bool(random.getrandbits(1))
        transform = random.choice([
            transforms.ColorJitter(brightness=rand_bright_num), transforms.ColorJitter(contrast=rand_cont_num),
            transforms.ColorJitter(saturation=rand_satur_num), transforms.ColorJitter(hue=rand_hue_num)])
        combined += [transform]
        """
    combined += [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    """
    if gaussian:
        combined += [AddGaussianNoise(0., 1.)]
    """
    transform = transforms.Compose(combined)
    return transform


def main():
    global opts, global_step
    opts = parser.parse_args()
    opts.cuda = 0

    global_step = 0

    print(opts)

    # Set GPU
    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        opts.cuda = 1
        print("Currently using GPU {}".format(opts.gpu_ids))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # Set model
    model = Res18_basic(NUM_CLASSES)
    model.eval()

    # set EMA model
    ema_model = Res18_basic(NUM_CLASSES)
    for param in ema_model.parameters():
        param.detach_()
    ema_model.eval()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if use_gpu:
        model.cuda()
        ema_model.cuda()

    model_for_test = ema_model  # change this to model if ema_model is not used.
    # model_for_test = model

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model_for_test)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################

    if opts.mode == 'train':
        # set multi-gpu
        if len(opts.gpu_ids.split(',')) > 1:
            model = nn.DataParallel(model)
            ema_model = nn.DataParallel(ema_model)
        model.train()
        ema_model.train()

        # Set dataloader

        # we will not change validation loader.
        # We will try to enhance the accuracy by changing only train_loader
        train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
        print(
            'found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))

        validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.CenterCrop(opts.imsize),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])),
            batch_size=opts.valbatchsize, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
        print('validation_loader done')

        # Set optimizer
        optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=5e-4)
        # optimizer = optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=5e-4)
        ema_optimizer = WeightEMA(model, ema_model)

        # INSTANTIATE LOSS CLASS
        train_criterion = SemiLoss()

        # INSTANTIATE STEP LEARNING SCHEDULER CLASS
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[50, 150], gamma=0.1)

        # Train and Validation 
        best_acc = -1
        # Time Signal Annealing in step-batchsize-decrease

        # Prior to baseline code which initialized loaders only at the start of the training,
        # We have changed the code so that it could load the loaders at every epochs
        # and also change the transform types(whether randomized or n = 1~4) for every epochs
        for epoch in range(opts.start_epoch, opts.epochs + 1):
            # we will give 2 drops for Time Signal Annealing
            # (different batchsize of labeled vs unlabled data for different epoch segments)
            if epoch > opts.epochdrop:
                # after first batchsize adjustment
                # returns transform through the function TransformType() defined above
                transform = TransformType(imResize=opts.imResize, imsize=opts.imsize, n=opts.n,
                                          randomize=opts.randomize,
                                          resize_crop=opts.resize_crop, gray=opts.gray, horizontal=opts.horizontal,
                                          vertical=opts.vertical, jitter=opts.jitter, rotate=opts.rotate,
                                          affine=opts.affine)

                # Rest of the contents for the loader is same except the transform is defined by
                # the function TransformType() above, and new parameter called k=opts.no_trainaug
                # is introduced. it is to change the train augmentation to enhance the number of data
                # the parameters here wil lbe sent to the ImageDataLoader_MT.py to be further used.
                train_loader = torch.utils.data.DataLoader(
                    SimpleImageLoader(DATASET_PATH, 'train', train_ids, k=opts.no_trainaug,
                                      transform=transform),
                    batch_size=opts.tbs_d, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

                unlabel_loader = torch.utils.data.DataLoader(
                    SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                                      transform=transform),
                    batch_size=opts.utb_d, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
            elif epoch > opts.epochdropdrop:
                # after second batchsize adjustment
                transform = TransformType(imResize=opts.imResize, imsize=opts.imsize, n=opts.n,
                                          randomize=opts.randomize,
                                          resize_crop=opts.resize_crop, gray=opts.gray, horizontal=opts.horizontal,
                                          vertical=opts.vertical, jitter=opts.jitter, rotate=opts.rotate,
                                          affine=opts.affine)


                train_loader = torch.utils.data.DataLoader(
                    SimpleImageLoader(DATASET_PATH, 'train', train_ids, k=opts.no_trainaug,
                                      transform=transform),
                    batch_size=opts.tbs_dd, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

                unlabel_loader = torch.utils.data.DataLoader(
                    SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                                      transform=transform),
                    batch_size=opts.utb_dd, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
            else:
                # before first batchsize adjustment
                transform = TransformType(imResize=opts.imResize, imsize=opts.imsize, n=opts.n,
                                          randomize=opts.randomize,
                                          resize_crop=opts.resize_crop, gray=opts.gray, horizontal=opts.horizontal,
                                          vertical=opts.vertical, jitter=opts.jitter, rotate=opts.rotate,
                                          affine=opts.affine)
                train_loader = torch.utils.data.DataLoader(
                    SimpleImageLoader(DATASET_PATH, 'train', train_ids, k=opts.no_trainaug,
                                      transform=transform),
                    batch_size=opts.batchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

                unlabel_loader = torch.utils.data.DataLoader(
                    SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                                      transform=transform),
                    batch_size=opts.unlbatchsize, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

            # print('start training')
            loss, loss_x, loss_u, avg_top1, avg_top5 = train(opts, train_loader, unlabel_loader, model, ema_model,
                                                             train_criterion, optimizer, ema_optimizer, epoch, use_gpu)
            # print('epoch {:03d}/{:03d} finished, loss: {:.3f}, loss_x: {:.3f}, loss_un: {:.3f}, avg_top1: {:.3f}%, avg_top5: {:.3f}%'.format(epoch, opts.epochs, loss, loss_x, loss_u, avg_top1, avg_top5))
            # scheduler.step()

            # print('start validation')
            acc_top1, acc_top5 = validation(opts, validation_loader, ema_model, epoch, use_gpu)
            print(
                'epoch {:03d}, loss: {:.3f}, loss_class: {:.3f}, loss_cons: {:.3f}, avg_top1: {:.3f}%, avg_top5: {:.3f}%, val_top1: {:.3f}%, val_top5: {:.3f}%'.format(
                    epoch, loss, loss_x, loss_u, avg_top1, avg_top5, acc_top1, acc_top5))
            is_best = acc_top1 > best_acc
            best_acc = max(acc_top1, best_acc)
            if is_best:
                print('model achieved the best accuracy ({:.3f}%) - saving best checkpoint...'.format(best_acc))
                if IS_ON_NSML:
                    nsml.save(opts.name + '_best')
                else:
                    torch.save(ema_model.state_dict(), os.path.join('runs', opts.name + '_best'))
            if (epoch + 1) % opts.save_epoch == 0:
                if IS_ON_NSML:
                    nsml.save(opts.name + '_e{}'.format(epoch))
                else:
                    torch.save(ema_model.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(epoch)))


def train(opts, train_loader, unlabel_loader, model, ema_model, criterion, optimizer, ema_optimizer, epoch, use_gpu):
    global global_step

    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_un = AverageMeter()

    losses_curr = AverageMeter()
    losses_x_curr = AverageMeter()
    losses_un_curr = AverageMeter()

    weight_scale = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    model.train()
    no_ta = opts.no_trainaug
    imsize = opts.imsize

    # nCnt =0 
    out = False
    local_step = 0
    while not out:
        labeled_train_iter = iter(train_loader)
        unlabeled_train_iter = iter(unlabel_loader)
        for batch_idx in range(len(train_loader)):
            try:
                data = labeled_train_iter.next()
                inputs_x1, inputs_x2, targets_x = data # import no_trainaug x 2 differently augmented training data. x2 is for obtaining consistency loss
            except:
                labeled_train_iter = iter(train_loader)
                data = labeled_train_iter.next()
                inputs_x1, inputs_x2, targets_x = data
            try:
                data = unlabeled_train_iter.next()
                inputs_u1, inputs_u2 = data # import 2 differently augmented unlabeled data
            except:
                unlabeled_train_iter = iter(unlabel_loader)
                data = unlabeled_train_iter.next()
                inputs_u1, inputs_u2 = data

            # Reshape train data tensors
            batch_size = targets_x.size(0)
            # print("\tinputs_x1 SIZE = \t", inputs_x1.size())  # [batchsize, no_augment, 3, 224, 224]
            # print("\ttargets_x SIZE = \t", targets_x.size())  # [batchsize, no_augment]
            inputs_x1 = torch.reshape(inputs_x1, (-1, 3, imsize, imsize)) # reshape train data to reduce new dimension added due to train data augmentation
            inputs_x2 = torch.reshape(inputs_x2, (-1, 3, imsize, imsize))
            targets_x = torch.reshape(targets_x, (-1, 1))
            # print("\tinputs_x1 SIZE = \t", inputs_x1.size())  # [batchsize*no_augment, 3, 224, 224]
            # print("\ttargets_x SIZE = \t", targets_x.size())  # [batchsize*no_augment, 1]

            # Transform label to one-hot
            classno = NUM_CLASSES
            targets_org = targets_x
            targets_x = torch.zeros(batch_size * no_ta, classno).scatter_(1, targets_x.view(-1, 1), 1)

            if use_gpu:
                inputs_x1, inputs_x2, targets_x = inputs_x1.cuda(), inputs_x2.cuda(), targets_x.cuda()
                inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()

            optimizer.zero_grad()

            #get prediction for student models
            fea, logits_x1 = model(inputs_x1)
            fea, logits_u1 = model(inputs_u1)

            #get predictions for teacher models
            fea, logits_x2 = ema_model(inputs_x2)
            fea, logits_u2 = ema_model(inputs_u2)

            # put interleaved samples back
            loss_class, loss_con, weigts_mixing = criterion(logits_x1, logits_x2, targets_x,
                                                            logits_u1, logits_u2,
                                                            epoch + batch_idx / len(train_loader), opts.epochs)
            loss = loss_class + opts.lambda_u * weigts_mixing * loss_con
            if weigts_mixing < 1:
                alpha = 0.99
            else:
                alpha = 0.999

            losses.update(loss.item(), inputs_x1.size(0))
            losses_x.update(loss_class.item(), inputs_x1.size(0))
            losses_un.update(loss_con.item(), inputs_x1.size(0))
            weight_scale.update(weigts_mixing, inputs_x1.size(0))

            losses_curr.update(loss.item(), inputs_x1.size(0))
            losses_x_curr.update(loss_class.item(), inputs_x1.size(0))
            losses_un_curr.update(loss_con.item(), inputs_x1.size(0))

            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
            ema_optimizer.step(alpha)

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                embed_x, pred_x1 = model(inputs_x1)

            if IS_ON_NSML and global_step % opts.log_interval == 0:
                nsml.report(step=global_step, loss=losses_curr.avg, loss_x=losses_x_curr.avg,
                            loss_un=losses_un_curr.avg)
                losses_curr.reset()
                losses_x_curr.reset()
                losses_un_curr.reset()

            acc_top1b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=1) * 100
            acc_top5b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=5) * 100
            acc_top1.update(torch.as_tensor(acc_top1b), inputs_x1.size(0))
            acc_top5.update(torch.as_tensor(acc_top5b), inputs_x1.size(0))

            local_step += 1
            global_step += 1

            if local_step >= opts.steps_per_epoch:
                out = True
                break

    return losses.avg, losses_x.avg, losses_un.avg, acc_top1.avg, acc_top5.avg



# validatation code same as prior baseline code given

def validation(opts, validation_loader, model, epoch, use_gpu):
    model.eval()
    avg_top1 = 0.0
    avg_top5 = 0.0
    nCnt = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
            nCnt += 1
            embed_fea, preds = model(inputs)

            acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1) * 100
            acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5) * 100
            avg_top1 += acc_top1
            avg_top5 += acc_top5

        avg_top1 = float(avg_top1 / nCnt)
        avg_top5 = float(avg_top5 / nCnt)

    if IS_ON_NSML:
        nsml.report(step=epoch, avg_top1=avg_top1, avg_top5=avg_top5)

    return avg_top1, avg_top5


if __name__ == '__main__':
    main()
