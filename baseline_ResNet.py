import sys
import os
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from modules.resnet import BasicBlock
from modules.resnet import ResNet
from utils import progress_bar
from utils import format_time
from dataLoader import dataloader4img

import argparse
from tensorboardX import SummaryWriter


def train(args, net, train_loader, optimizer, scheduler, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    begin_time = time.time()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        current_time = time.time()
        step_time = current_time - begin_time
        if args.show != 0:
            progress_bar(step_time, batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        batch_idx += 1
    loss_ = train_loss / (batch_idx + 1)
    acc_ = correct / total
    scheduler.step()
    return loss_, acc_


def test(args, net, test_loader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    begin_time = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            current_time = time.time()
            step_time = current_time - begin_time

            if args.show != 0:
                progress_bar(step_time, batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            batch_idx += 1
    loss_ = test_loss / (batch_idx + 1)
    acc_ = correct / total
    return loss_, acc_


def main(args):
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    os.makedirs(args.tensorboard_path, exist_ok=True)
    writer = SummaryWriter(logdir=args.tensorboard_path)

    train_set, test_set = dataloader4img(args=args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

    model = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], img_size=args.img_size, num_classes=args.num_classes)
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # train & test
    best_test_acc = 0
    start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train(args=args,
                                      net=model,
                                      train_loader=train_loader,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      criterion=criterion)
        test_loss, test_acc = test(args=args,
                                   net=model,
                                   test_loader=test_loader,
                                   criterion=criterion)

        # update the best test acc & save the best model
        if test_acc > best_test_acc:
            state = {
                'net': model.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if args.checkpoint_path:
                os.makedirs(args.checkpoint_path, exist_ok=True)
                checkpoint_path = os.path.join(args.checkpoint_path, 'model.pth')
                torch.save(state, checkpoint_path)
            best_test_acc = test_acc

        # log in tensorboard
        writer.add_scalar('train/accuracy', scalar_value=train_acc, global_step=epoch)
        writer.add_scalar('train/loss', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar('train/lr', scalar_value=optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        writer.add_scalar('test/accuracy', scalar_value=test_acc, global_step=epoch)
        writer.add_scalar('test/loss', scalar_value=test_loss, global_step=epoch)

    writer.close()
    end = time.time()
    total_time = format_time(end-start)
    print("Total time:", total_time)
    print("The best test acc is", best_test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training models for image classification')
    parser.add_argument('--path', default='./data/main', type=str, help='data path contains train and test')
    parser.add_argument('--checkpoint_path', default='./checkpoint/resnet', type=str, help='checkpoint path')
    parser.add_argument('--tensorboard_path', default='./tensorboard/resnet', type=str, help='tensorboard path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num_classes', default=12, type=int, help='number of classes')
    parser.add_argument('--img_size', default=192, type=int, help='resize the images into img_size*img_size')
    parser.add_argument('--epochs', default=400, type=int, help='training epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=60, type=int, help='batch size')
    parser.add_argument('--step_size', default=50, type=int, help='step size for schedule')
    parser.add_argument('--gamma', default=0.5, type=int, help='gamma for schedule')
    parser.add_argument('--n_workers', default=0, type=int, help='train model on multiple gpus')
    parser.add_argument('--show', default=0, type=int, help='whether use the progress bar')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    args = parser.parse_args()
    main(args)
