import os
import sys
import importlib.util
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from models_mopdo.networks import CONFIGS, MoPDOVisionTransformer
from datasets_mopdo.VTABDataLoader import get_data
from utils import (seed_torch, accuracy, AverageMeter, Logger, count_parameters)
from timm.scheduler import create_scheduler
from torch.cuda.amp import autocast
from timm.utils import NativeScaler
from timm.models import model_parameters
import argparse


def load_data_configs(config_file):
    config_path = os.path.abspath(config_file)
    spec = importlib.util.spec_from_file_location("vtab_config_override", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load config file: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DATA_CONFIGS


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", default="MoPDO training")
    parser.add_argument("--dataset_name", default="cifar")
    parser.add_argument("--config_file", default="./datasets_mopdo/Explore_VTABConfig.py")
    parser.add_argument("--model_type", default="ViT-B_16")
    parser.add_argument("--dataset_dir", default="./data/vtab-1k")
    parser.add_argument("--pretrained_dir", type=str, default="ViT-B_16.npz")
    parser.add_argument("--output_dir", default="output/vtab_mopdo", type=str)
    parser.add_argument("--device", default='cuda', type=str)

    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--num_classes", default=100, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=3e-3, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--simple_aug", default=True, type=bool)
    parser.add_argument("--entropy_scale", default=None, type=float)
    parser.add_argument("--drop_path", default=None, type=float)
    parser.add_argument("--max_train_batches", default=None, type=int)
    parser.add_argument("--max_eval_batches", default=None, type=int)
    parser.add_argument("--max_loop_epochs", default=None, type=int)
    
    # fellow SSF
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--sched", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--lr_cycle_decay", default=0.5, type=float)
    parser.add_argument("--cooldown_epochs", default=10, type=int)

    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--loss_scale', type=float, default=0)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    return args


def frozen_param(model, frozen_list=('',)):
    for name, param in model.named_parameters():
        if any(item in name for item in frozen_list):
            param.requires_grad = True
        else:
            param.requires_grad = False
    num_params = count_parameters(model)
    print("Training parameters %s", args)
    print("Total Parameter: \t%2.3fM" % num_params)


def save_model(max_acc, acc, model, path):
    if acc > max_acc:
        from collections import OrderedDict
        model_dict = OrderedDict()
        save_index = ['head', 'scale', 'shift', 'hco']
        for k, v in model.state_dict().items():
            if any(item in k for item in save_index):
                model_dict[k] = v
        if os.path.exists(path + '{:6.2f}'.format(max_acc) + '.pth'):
            os.remove(path + '{:6.2f}'.format(max_acc) + '.pth')
        torch.save(model_dict, path + '{:6.2f}'.format(acc) + '.pth')
        return acc
    return max_acc


def setup(args, frozen_list=('',)):
    # Prepare model
    config = CONFIGS[args.model_type]
    model = MoPDOVisionTransformer(config, args.img_size, zero_head=True, num_classes=args.num_classes, drop_path=args.drop_path)
    model.load_from(np.load(args.pretrained_dir))

    frozen_param(model, frozen_list)
    return model


@torch.no_grad()
def valid(model, test_loader, device, max_batches=None):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (x, label) in enumerate(tqdm(test_loader)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x, label = x.to(device), label.to(device)
        with autocast():
            output = model(x)

        loss = criterion(output, label)
        acc1 = accuracy(output, label, topk=(1,))
        top1.update(acc1[0].item(), x.size(0))
        losses.update(loss.item(), x.size(0))
    print('Test :', losses, top1)
    return top1.avg, losses.avg


def train(model, train_loader, criterion, optimizer, loss_scaler, lr_scheduler, epoch, entropy_scale, max_epoch, max_batches=None):
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data, target = data.to(device), target.long().to(device)
        with autocast():
            output, loss_reg = model(
                data,
                is_training=True,
                epoch=epoch,
                max_epoch=max_epoch,
                entropy_scale=entropy_scale,
            )
            loss_cls = criterion(output, target)
            loss = loss_cls + loss_reg

        optimizer.zero_grad()
        # fellow SSF
        loss_scaler(loss, optimizer, parameters=model_parameters(model))
        # loss.backward()
        # optimizer.step()

        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0].item(), data.size(0))

    # fellow SSF
    if lr_scheduler is not None:
        lr_scheduler.step_update(num_updates=epoch, metric=losses.avg)
    print('Train :', losses, top1)
    return top1.avg, losses.avg


def main(args):
    data_configs = load_data_configs(args.config_file)
    config = data_configs[args.dataset_name]
    args.data_path = os.path.join(args.dataset_dir, args.dataset_name)
    args.num_classes = config['num_classes']
    args.learning_rate = config['lr']
    args.min_lr = config['min_lr']
    if args.drop_path is None:
        args.drop_path = config['drop_path']
    args.warmup_lr = config['warmup_lr']
    args.weight_decay = config['weight_decay']
    args.batch_size = config['batch_size']
    args.simple_aug = config['simple_aug']
    if args.entropy_scale is None:
        args.entropy_scale = config['entropy_scale']
    if not os.path.exists(os.path.join(args.output_dir, args.dataset_name)):
        os.makedirs(os.path.join(args.output_dir, args.dataset_name))
    else:
        import shutil
        shutil.rmtree(os.path.join(args.output_dir, args.dataset_name))
        os.makedirs(os.path.join(args.output_dir, args.dataset_name))
    sys.stdout = Logger(sys.stdout, os.path.join(args.output_dir, args.dataset_name, '{}.txt').format(args.dataset_name))

    train_loader, test_loader = get_data(
        data_path=args.data_path,
        batch_size=args.batch_size,
        simple_aug=args.simple_aug,
        num_workers=config['num_workers'],
    )


    model = setup(args, ['head', 'scale', 'shift', 'hco'])
    model.to(device)

    max_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()
    optimizer = torch.optim.AdamW(model.get_parameters(lr=args.learning_rate, weight_decay=args.weight_decay))   #
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    print(args, config)
    loop_epochs = num_epochs if args.max_loop_epochs is None else min(num_epochs, args.max_loop_epochs)
    for epoch in range(0, loop_epochs):
        train(
            model,
            train_loader,
            criterion,
            optimizer,
            loss_scaler,
            lr_scheduler,
            epoch,
            args.entropy_scale,
            num_epochs,
            args.max_train_batches,
        )
        # fellow SSF
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        if epoch % 1 == 0 and epoch > 0:
            acc, _ = valid(model, test_loader, device, args.max_eval_batches)
            max_acc = save_model(max_acc, acc, model, os.path.join(args.output_dir, args.dataset_name, args.dataset_name))
            print('Epoch {}, cur_max_acc : {}'.format(epoch, max_acc))
            current_lr = optimizer.param_groups[0]['lr']
            print(f"cur lr: {current_lr}")
            

if __name__ == '__main__':
    args = get_args_parser()
    seed_torch(args.seed)
    device = torch.device(args.device)
    main(args)
