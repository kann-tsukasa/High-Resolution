import sys
import os
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import GraphDataLoader

from modules.gin import GIN
from utils import progress_bar
from utils import format_time

from tensorboardX import SummaryWriter
import argparse


def train(args, net, train_loader, optimizer, scheduler, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    begin_time = time.time()
    for graphs, labels in train_loader:
        # batch graphs will be shipped to device in forward part of model
        graphs, labels = graphs.to(args.device), labels.to(args.device)
        feat = graphs.ndata.pop('feats')
        optimizer.zero_grad()
        try:
            outputs = net(graphs, feat)
        except:
            continue
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
        for graphs, labels in test_loader:
            graphs, labels = graphs.to(args.device), labels.to(args.device)
            feat = graphs.ndata.pop('feats')
            outputs = net(graphs, feat)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            current_time = time.time()
            step_time = current_time - begin_time
            if args.show != 0:
                progress_bar(step_time, batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    loss_ = test_loss / (batch_idx + 1)
    acc_ = correct / total
    return loss_, acc_


def main(args):
    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    writer = SummaryWriter(logdir=args.tensorboard_path)

    train_dict = list(np.load('./data/graphs/train.npy', allow_pickle=True))
    test_dict = list(np.load('./data/graphs/test.npy', allow_pickle=True))
    train_data = []
    test_data = []
    label_set = set()
    while train_dict:
        temp = train_dict.pop()
        train_data.append((temp[0], temp[1]))
        label_set.add(temp[1])
    while test_dict:
        temp = test_dict.pop()
        test_data.append((temp[0], temp[1]))
    args.n_feats = len(train_data[0][0].ndata['feats'][0])
    args.num_classes = len(label_set)
    train_loader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = GraphDataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = GIN(num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers,
                input_dim=args.n_feats, hidden_dim=args.hidden_dim, output_dim=args.num_classes,
                final_dropout=args.final_dropout, learn_eps=args.learn_eps,
                graph_pooling_type=args.graph_pooling_type,
                neighbor_pooling_type=args.neighbor_pooling_type)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(args.tensorboard_path, exist_ok=True)
    writer = SummaryWriter(logdir=args.tensorboard_path)

    best_test_acc = 0
    start = time.time()
    for epoch in range(args.epochs):
        print("Epoch:", epoch+1)
        print("Training")
        train_loss, train_acc = train(args=args,
                                      net=model,
                                      train_loader=train_loader,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      criterion=criterion)
        print("Testing")
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
    total_time = format_time(end - start)
    print("Total time:", total_time)
    print("The best test acc is", best_test_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training models for image classification')
    parser.add_argument('--checkpoint_path', default='./checkpoint/gbcm_np', type=str, help='checkpoint path')
    parser.add_argument('--tensorboard_path', default='./tensorboard/gbcm_np', type=str, help='tensorboard path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--n_feats', default=988, type=int, help='feature size')
    parser.add_argument('--num_classes', default=12, type=int, help='number of classes')
    parser.add_argument('--num_layers', type=int, default=5, help='number of layers (default: 5)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of MLP layers(default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='number of hidden units (default: 128)')
    parser.add_argument('--final_dropout', type=float, default=0.5, help='final layer dropout (default: 0.5)')
    parser.add_argument('--learn_eps', action="store_true", help='learn the epsilon weighting')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "mean", "max"],
                        help='type of graph pooling: sum, mean or max')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "mean", "max"],
                        help='type of neighboring pooling: sum, mean or max')

    # parser.add_argument('--img_size', default=224, type=int, help='resize the images into img_size*img_size')
    parser.add_argument('--epochs', default=400, type=int, help='training epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--step_size', default=50, type=int, help='step size for schedule')
    parser.add_argument('--gamma', default=0.8, type=int, help='gamma for schedule')
    parser.add_argument('--n_workers', default=0, type=int, help='train model on multiple gpus')
    parser.add_argument('--show', default=0, type=int, help='whether use the progress bar')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    args = parser.parse_args()
    main(args)
