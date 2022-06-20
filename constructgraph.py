import os
from tqdm import tqdm

import pandas as pd
import numpy as np

import cv2
from PIL import Image

from extractor import *

import torch
import dgl


class ConstructGraph(object):
    """根据图片数据构图

    """
    def __init__(self, edge_path, img_path, dict_only=False, test_only=False, nfeat_type='embed'):
        # 节点特征类型
        self.nfeat_type = nfeat_type
        # 数据读取路径
        self.edge_path = edge_path
        self.img_path = img_path
        # 边的数据，还是文件名形式
        self.edge = None
        self.n_edge = None
        # 存储节点的特征（3*64*64）
        self.features = []
        # 词典，用于索引图片和图片编号
        self.idx_dict = {}
        self.name_dict = {}
        # 训练测试mask
        self.train_mask = []
        self.test_mask = []
        # 图的边
        self.from_list = []
        self.to_list = []
        # 标签
        self.labels = []
        self.label_dict = {}
        # 图
        self.g = None
        # 是否只返回图片&图中编号的词典
        self.dict_only = dict_only
        # 是否只使用测试集数据构图
        self.test_only = test_only

    def read_edges(self):
        edge = pd.read_csv(self.edge_path)
        self.edge = edge
        self.n_edge = self.edge.shape[0]
        print("Edge info loaded.")

    def construct_info(self):
        n = self.n_edge
        from_list = list(self.edge.iloc[:, 0])
        to_list = list(self.edge.iloc[:, 1])
        idx = 0
        label_idx = 0
        if self.nfeat_type == 'embed':
            d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            feature_generator = Model(opt)
            feature_generator = torch.nn.DataParallel(feature_generator).to(d)
            # load model
            model_path = './backbone/TPS-ResNet-BiLSTM-Attn.pth'
            feature_generator.load_state_dict(torch.load(model_path, map_location=d))
        print('Start constructing node features.')
        for _ in tqdm(range(n)):
            _f, _t = from_list[_], to_list[_]

            # 记录train_mask和test_mask
            mk_f, mk_t = _f.split('_')[-1], _t.split('_')[-1]
            # 若只要测试集的图
            if self.test_only and (mk_f == 'train' or mk_t == 'train'):
                continue

            # 为图片编号，并制作kv索引，对应回原数据

            if _f not in self.name_dict:
                # 更新用于用户编号&图片名双向索引的两个词典
                self.idx_dict[idx] = _f
                self.name_dict[_f] = idx
                self.from_list.append(idx)
                # 若只需要词典，就不需要保存特征
                if not self.dict_only:
                    # 第一次遇到该节点，读取图片作为特征
                    img_path = os.path.join(self.img_path, f'{_f}.jpg')
                    if self.nfeat_type == 'img':
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = np.array([img[:, :, _] for _ in range(3)])/255
                        img = np.array(img, dtype=np.float32)
                        self.features.append(img)
                    elif self.nfeat_type == 'embed':
                        image = Image.open(img_path).convert('L')
                        image = image.resize((100, 32), Image.BICUBIC)
                        import torchvision.transforms as trans
                        trans = trans.ToTensor()
                        image = np.array([np.array(trans(image).sub_(0.5).div_(0.5))])
                        feature = f_inference(model=feature_generator, input=torch.tensor(image), device=d)
                        self.features.append(np.array(feature.cpu(), dtype=np.float32))
                    else:
                        raise Exception('Unknown nfeat_type')

                if mk_f == 'train':
                    self.train_mask.append(True)
                    self.test_mask.append(False)
                else:
                    self.train_mask.append(False)
                    self.test_mask.append(True)

                # 获取样本的label
                label = _f.split('_')[1]
                if label not in self.label_dict:
                    self.label_dict[label] = label_idx
                    self.labels.append(label_idx)
                    label_idx += 1
                else:
                    self.labels.append(self.label_dict[label])
                idx += 1
            else:
                # 否则，只需获取它的编号
                self.from_list.append(self.name_dict[_f])

            if _t not in self.name_dict:
                # 更新用于用户编号&图片名双向索引的两个词典
                self.idx_dict[idx] = _t
                self.name_dict[_t] = idx
                self.to_list.append(idx)
                # 若只需要词典，就不需要保存特征
                if not self.dict_only:
                    # 第一次遇到该节点，读取图片作为特征
                    img2_path = os.path.join(self.img_path, f'{_t}.jpg')
                    if self.nfeat_type == 'img':
                        img2 = cv2.imread(img2_path)
                        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                        img2 = np.array([img2[:, :, _] for _ in range(3)]) / 255
                        img2 = np.array(img2, dtype=np.float32)
                        self.features.append(img2)
                    elif self.nfeat_type == 'embed':
                        image = Image.open(img2_path).convert('L')
                        image = image.resize((100, 32), Image.BICUBIC)
                        import torchvision.transforms as trans
                        trans = trans.ToTensor()
                        image = np.array([np.array(trans(image).sub_(0.5).div_(0.5))])
                        feature = f_inference(model=feature_generator, input=torch.tensor(image), device=d)
                        self.features.append(np.array(feature.cpu(), dtype=np.float32))
                if mk_t == 'train':
                    self.train_mask.append(True)
                    self.test_mask.append(False)
                else:
                    self.train_mask.append(False)
                    self.test_mask.append(True)

                # 获取样本的label
                label = _f.split('_')[1]
                if label not in self.label_dict:
                    self.label_dict[label] = label_idx
                    self.labels.append(label_idx)
                    label_idx += 1
                else:
                    self.labels.append(self.label_dict[label])
                idx += 1
            else:
                # 否则，只需获取它的编号
                self.to_list.append(self.name_dict[_t])

    def construct_graph(self):
        self.read_edges()
        self.construct_info()
        g = dgl.graph((self.from_list + self.to_list, self.to_list + self.from_list))
        g = dgl.add_self_loop(g)
        # print(len(self.train_mask))
        # print(len(self.test_mask))
        # print(g.num_nodes())
        # print(len(self.features))
        if not self.dict_only:
            g.ndata['features'] = torch.tensor(np.array(self.features, dtype=np.float32), dtype=torch.float32)
        g.ndata['train_mask'] = torch.tensor(self.train_mask)
        g.ndata['test_mask'] = torch.tensor(self.test_mask)
        g.ndata['labels'] = torch.tensor(self.labels)
        self.g = g


if __name__ == '__main__':
    edge_path = './data/dicts/edges.csv'
    img_path = './data/subimgs'
    cons_graph = ConstructGraph(edge_path=edge_path, img_path=img_path, nfeat_type='img')  # img or embed
    cons_graph.construct_graph()
    dgl.data.utils.save_info('./data/graph/graph.bin', cons_graph.g)
    idx_dict = cons_graph.idx_dict
    name_dict = cons_graph.name_dict
    label_dict = cons_graph.label_dict
    np.save('./data/dicts/idx_dict.npy', idx_dict)
    np.save('./data/dicts/name_dict.npy', name_dict)
    np.save('./data/dicts/label_dict.npy', label_dict)


