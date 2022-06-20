from modules.nbcm import *
import argparse
import torch as th
import time
from tensorboardX import SummaryWriter
import os
import datetime
import numpy as np
from torchvision.models import resnet18


def get_acc(d):
    d2 = {}
    for k, v in d.items():
        ori = '_'.join(k.split('_')[:2])
        if ori not in d2:
            d2[ori] = []
        d2[ori].append(v)
    c3 = 0
    for k, v in d2.items():
        car = k.split('_')[-1]
        x = {}
        others = 0
        for _ in v:
            if _ not in x:
                x[_] = 0
            x[_] += 1
            if _ == 'others':
                others += 1
        if others/len(v) == 1:
            r = 'others'
        else:
            try:
                x.pop('others')
            except:
                pass
        r = max(x)
        if car == r:
            c3 += 1
    return c3/len(d2)


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def run(args, g, device, dict_path):
    # 记录数据
    now = datetime.datetime.timestamp(datetime.datetime.now())
    if args.pretrain == 1:
        tensorboard_dir = './tensorboard/nbcm_p'
    else:
        tensorboard_dir = './tensorboard/nbcm_np'
    os.makedirs(tensorboard_dir, exist_ok=True)

    # g = dgl.data.utils.load_info('./graph_data.bin')
    train_g = g.subgraph(g.ndata['train_mask'])
    test_g = g.subgraph(g.ndata['test_mask'])
    # test_g = g

    n_classes = len(np.unique(np.array(g.ndata['labels'])))
    img_size = g.ndata['features'].shape[2]

    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]

    train_nfeat = train_g.ndata.pop('features')
    train_labels = train_g.ndata.pop('labels')

    test_nfeat = test_g.ndata.pop('features')
    test_labels = test_g.ndata.pop('labels')

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # 获取编号对应回原图片的字典
    idx_dict = np.load(os.path.join(dict_path, 'idx_dict.npy'), allow_pickle=True).item()
    name_dict = np.load(os.path.join(dict_path, 'name_dict.npy'), allow_pickle=True).item()
    label_dict = np.load(os.path.join(dict_path, 'label_dict.npy'), allow_pickle=True).item()

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    test_dataloader = dgl.dataloading.NodeDataLoader(
        test_g,
        test_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    vggraphsage = SAGE(img_size=img_size, n_hidden=args.num_hidden, n_classes=n_classes, n_layers=args.num_layers,
                       activation=F.relu, dropout=args.dropout)
    if args.pretrain == 1:
        m = resnet18(pretrained=True)
        for k, v in m.state_dict().items():
            if k in vggraphsage.state_dict():
                vggraphsage.state_dict()[k] = v

    vggraphsage = vggraphsage.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vggraphsage.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # optimizer = optim.SGD(vggraphsage.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    # Training loop
    avg = 0
    iter_tput = []
    writer = SummaryWriter(logdir=tensorboard_dir)

    for epoch in range(args.num_epochs):
        tic = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        # tic_step = time.time()
        train_acc = 0
        test_acc = 0
        best_acc = 0
        avg_acc = []
        loss_total_train = 0
        step = 0
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            optimizer.zero_grad()
            batch_pred = vggraphsage(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            loss_total_train += loss.item()
            loss.backward()
            optimizer.step()
            # iter_tput.append(len(seeds) / (time.time() - tic_step))
            # if step % args.log_every == 0:
            #     acc = compute_acc(batch_pred, batch_labels)
            #     gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
            #     if epoch == 0 and step == 0:
            #         speed = 0
            #     else:
            #         speed = np.mean(iter_tput[3:])
            #     train_acc = acc.item()
            #     print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(epoch, step, loss.item(), train_acc, speed, gpu_mem_alloc))
            # tic_step = time.time()
        train_loss = loss_total_train / (step + 1)
        # toc = time.time()
        # print('Epoch Time(s): {:.4f}'.format(toc - tic))
        # avg_acc.append(train_acc)
        # if epoch >= 5:
        #     avg += toc - tic
        # test_acc = evaluate(vggraphsage, test_g, test_nfeat, test_labels, device)
        # best_acc = max(test_acc, best_acc)
        # state = {'net': vggraphsage.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        # th.save(state, f"./checkpoint/vggraphsage_accu{best_acc}.pth")

        # 测试集LOSS
        loss_total_test = 0
        step = 0
        vggraphsage.eval()
        with torch.no_grad():
            for step, (input_nodes, seeds, blocks) in enumerate(test_dataloader):
                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(test_nfeat, test_labels, seeds, input_nodes, device)
                blocks = [block.int().to(device) for block in blocks]

                # Compute loss and prediction
                batch_pred = vggraphsage(blocks, batch_inputs)
                loss2 = loss_fcn(batch_pred, batch_labels)
                loss_total_test += loss2.item()

            test_loss = loss_total_test/(step+1)

        train_dict = give_results(model=vggraphsage, g=train_g, nfeat=train_nfeat, device=device, idx_dict=idx_dict, label_dict=label_dict)
        test_dict = give_results(model=vggraphsage, g=test_g, nfeat=test_nfeat, device=device, idx_dict=idx_dict, label_dict=label_dict)
        train_acc = get_acc(train_dict)
        test_acc = get_acc(test_dict)
        print(epoch, train_acc, test_acc)
        # np.save(f"./res_dict/res_dict_accu{best_acc}.npy", res_dict)
        writer.add_scalar('train/accuracy', scalar_value=train_acc, global_step=epoch)
        writer.add_scalar('train/loss', scalar_value=train_loss, global_step=epoch)
        writer.add_scalar('train/lr', scalar_value=optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        writer.add_scalar('test/accuracy', scalar_value=test_acc, global_step=epoch)
        writer.add_scalar('test/loss', scalar_value=test_loss, global_step=epoch)
        # print('Test Acc: {:.4f}'.format(test_acc))
        scheduler.step()
    writer.close()
    print('Avg epoch time: {}'.format(avg / (args.num_epochs - 5)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("training imagesage")
    argparser.add_argument('--gpu', type=int, default=0,  help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--pretrain', type=int, default=0, help='whether use a pretrained resnet18')
    argparser.add_argument('--feat_type', type=str, default='img', choices=['img', 'embed'])
    argparser.add_argument('--num_epochs', type=int, default=400)
    argparser.add_argument('--num_hidden', type=int, default=224)
    argparser.add_argument('--num_layers', type=int, default=2)
    argparser.add_argument('--fan_out', type=str, default='6,6')
    argparser.add_argument('--batch_size', type=int, default=50)
    argparser.add_argument('--log_every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num_workers', type=int, default=1,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    g = dgl.data.utils.load_info('./data/graph/graph.bin')
    # g = dgl.to_bidirected(g)
    dict_path = './data/dicts'
    run(args=args, g=g, device=device, dict_path=dict_path)
