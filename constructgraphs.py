import os
from tqdm import tqdm

import numpy as np
import pandas as pd
# import cv2
from PIL import Image

import dgl

from extractor import *


class ConstructGraphs(object):
    """根据图片构造图分类数据集
    :param debug_run: load n=debug_run images for test. If debug_run==0, then load all the dataset.
    """
    def __init__(self, data_path, debug_run=0, graph_method='bfs'):
        self.debug_run = debug_run
        self.data_path = data_path
        self.train_dataset = []
        self.test_dataset = []
        self.graph_dict = {}
        self.n_feats = None
        self.label_dict = {}
        self.graph_method = graph_method

    def get_relations(self):
        """
        """
        if self.graph_method == 'full':
            # get the image roots
            file_list = os.listdir(self.data_path)
            if '.DS_Store' in file_list:
                file_list.remove('.DS_Store')
            n = len(file_list)
            graph_list_dict = {}
            for i in tqdm(range(0, n)):
                file = file_list[i]
                graph_idx = int(file.split('_')[0])
                if graph_idx not in graph_list_dict:
                    graph_list_dict[graph_idx] = []
                graph_list_dict[graph_idx].append(file)
            graph_dict = {}
            for _ in tqdm(range(1, n+1)):
                img_list = graph_list_dict[_]
                # 构造全连接图的边关系
                m = len(img_list)
                i = 0
                pair_list = []
                while i < m:
                    j = i
                    while j < m:
                        pair_list.append([img_list[i], img_list[j]])
                        j += 1
                    i += 1
                graph_dict[_] = pair_list
            self.graph_dict = graph_dict

        elif self.graph_method == 'bfs':
            try:
                graph_dict = {}
                edges = pd.read_csv('./data/dicts/edges.csv')
                for _ in range(edges.shape[0]):
                    f_, t_ = edges.iloc[_, :]
                    graph_idx = int(f_.split('_')[0])  # get the original img name
                    if graph_idx not in graph_dict:
                        graph_dict[graph_idx] = []
                    graph_dict[graph_idx].append([f_, t_])
                    graph_dict[graph_idx].append([t_, f_])
                self.graph_dict = graph_dict
            except:
                raise Exception('missing edges data')
        else:
            raise Exception('unknown graph method')

    def construct_graphs(self, model_path=None):
        """

        """
        # transform
        import torchvision.transforms as trans
        trans = trans.ToTensor()
        # embedding
        if not model_path:
            model_path = './backbone/TPS-ResNet-BiLSTM-Attn.pth'
        d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_generator = Model(opt)
        feature_generator = torch.nn.DataParallel(feature_generator).to(d)
        # load model
        feature_generator.load_state_dict(torch.load(model_path, map_location=d))
        self.get_relations()
        cls = 0
        n = len(self.graph_dict)
        if self.debug_run > 0:
            n = int(self.debug_run)
        for _ in tqdm(self.graph_dict.keys()):
            pair_list = self.graph_dict[_]
            idx_dict = {}
            img_list = []
            idx = 0
            from_list = []
            to_list = []
            for pair in pair_list:
                f_, t_ = pair
                if f_ not in idx_dict:
                    idx_dict[f_] = idx
                    img_list.append(f_)
                    idx += 1
                if t_ not in idx_dict:
                    idx_dict[t_] = idx
                    img_list.append(t_)
                    idx += 1
                from_list.append(idx_dict[f_])
                to_list.append(idx_dict[t_])
            # 读取图片作为节点特征
            node_feats = []
            for img in img_list:
                if '.jpg' not in img:
                    img_path = os.path.join(self.data_path, f'{img}.jpg')
                else:
                    img_path = os.path.join(self.data_path, f'{img}')
                image = Image.open(img_path).convert('L')
                image = image.resize((100, 32), Image.BICUBIC)
                image = np.array([np.array(trans(image).sub_(0.5).div_(0.5))])
                feature = f_inference(model=feature_generator, input=torch.tensor(image), device=d)
                node_feats.append(np.array(feature.cpu()))
            # construct the graph
            g = dgl.graph((from_list, to_list))
            g = dgl.add_self_loop(g)
            g.ndata['feats'] = torch.tensor(np.array(node_feats), dtype=torch.float32)
            # 获取标签
            label = img_list[0].split('_')[1]
            if label not in self.label_dict:
                self.label_dict[label] = cls
                cls += 1
            graph_label = self.label_dict[label]
            # 切分训练集和测试集
            split_type = img_list[0].split('.')[0].split('_')[-1]
            if split_type == 'train':
                self.train_dataset.append((g, graph_label))
            else:
                self.test_dataset.append((g, graph_label))
        self.n_feats = len(feature)
        # for _ in tqdm(range(1, n+1)):
        #     img_list = self.graph_dict[_]
        #     # 构造全连接图的边关系
        #     m = len(img_list)
        #     i = 0
        #     from_list = []
        #     to_list = []
        #     while i < m:
        #         j = i
        #         while j < m:
        #             from_list.append(i)
        #             to_list.append(j)
        #             j += 1
        #         i += 1
        #     # 读取图片作为节点特征
        #     node_feats = []
        #     for img in img_list:
        #         img_path = os.path.join(self.data_path, f'{img}')
        #         image = Image.open(img_path).convert('L')
        #         image = image.resize((100, 32), Image.BICUBIC)
        #         image = np.array([np.array(trans(image).sub_(0.5).div_(0.5))])
        #         # image = cv2.imread(img_path)
        #         # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #         # image = cv2.resize(image, (100, 32), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
        #         # image = np.array([image[:, :, _] for _ in range(3)]) / 255
        #         feature = f_inference(model=feature_generator, input=torch.tensor(image), device=d)
        #         node_feats.append(np.array(feature.cpu()))
        #     # 构造图
        #     g = dgl.graph((from_list, to_list))
        #     g.ndata['feats'] = torch.tensor(np.array(node_feats), dtype=torch.float32)
        #     # 获取标签
        #     label = img_list[0].split('_')[1]
        #     if label not in self.label_dict:
        #         self.label_dict[label] = cls
        #         cls += 1
        #     graph_label = self.label_dict[label]
        #     # 切分训练集和测试集
        #     split_type = img_list[0].split('.')[0].split('_')[-1]
        #     if split_type == 'train':
        #         self.train_dataset.append((g, graph_label))
        #     else:
        #         self.test_dataset.append((g, graph_label))
        # self.n_feats = len(feature)


if __name__ == '__main__':
    # loader = ConstructGraphs(data_path='./dataset/subimgs')
    # loader.construct_graphs()
    # np.save('./dataset/fdata/train.npy', loader.train_dataset)
    # np.save('./dataset/fdata/test.npy', loader.test_dataset)

    loader = ConstructGraphs(data_path='./data/subimgs', graph_method='bfs')
    loader.construct_graphs()
    os.makedirs("./data/graphs")
    np.save('./data/graphs/train.npy', loader.train_dataset)
    np.save('./data/graphs/test.npy', loader.test_dataset)






# import os
# from tqdm import tqdm
#
# import numpy as np
# # import cv2
# from PIL import Image
#
# import dgl
#
# from extractor import *
#
#
# class ConstructGraphs(object):
#     """根据图片构造图分类数据集
#     :param debug_run: load n=debug_run images for test. If debug_run==0, then load all the dataset.
#     """
#     def __init__(self, data_path, debug_run=0):
#         self.debug_run = debug_run
#         self.data_path = data_path
#         self.train_dataset = []
#         self.test_dataset = []
#         self.graph_dict = {}
#         self.n_feats = None
#         self.label_dict = {}
#
#     def get_relations(self):
#         """
#         """
#         # get the image roots
#         file_list = os.listdir(self.data_path)
#         if '.DS_Store' in file_list:
#             file_list.remove('.DS_Store')
#         n = len(file_list)
#         graph_dict = {}
#         for i in tqdm(range(0, n)):
#             file = file_list[i]
#             graph_idx = int(file.split('_')[0])
#             if graph_idx not in graph_dict:
#                 graph_dict[graph_idx] = []
#             graph_dict[graph_idx].append(file)
#         self.graph_dict = graph_dict
#
#     def construct_graphs(self, model_path=None):
#         """
#
#         """
#         # transform
#         import torchvision.transforms as trans
#         trans = trans.ToTensor()
#         # embedding
#         if not model_path:
#             model_path = './backbone/TPS-ResNet-BiLSTM-Attn.pth'
#         d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         feature_generator = Model(opt)
#         feature_generator = torch.nn.DataParallel(feature_generator).to(d)
#         # load model
#         feature_generator.load_state_dict(torch.load(model_path, map_location=d))
#
#         self.get_relations()
#         cls = 0
#         n = len(self.graph_dict)
#         if self.debug_run > 0:
#             n = int(self.debug_run)
#         for _ in tqdm(range(1, n+1)):
#             img_list = self.graph_dict[_]
#             # 构造全连接图的边关系
#             m = len(img_list)
#             i = 0
#             from_list = []
#             to_list = []
#             while i < m:
#                 j = i
#                 while j < m:
#                     from_list.append(i)
#                     to_list.append(j)
#                     j += 1
#                 i += 1
#             # 读取图片作为节点特征
#             node_feats = []
#             for img in img_list:
#                 img_path = os.path.join(self.data_path, f'{img}')
#                 image = Image.open(img_path).convert('L')
#                 image = image.resize((100, 32), Image.BICUBIC)
#                 image = np.array([np.array(trans(image).sub_(0.5).div_(0.5))])
#                 # image = cv2.imread(img_path)
#                 # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 # image = cv2.resize(image, (100, 32), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
#                 # image = np.array([image[:, :, _] for _ in range(3)]) / 255
#                 feature = f_inference(model=feature_generator, input=torch.tensor(image), device=d)
#                 node_feats.append(np.array(feature.cpu()))
#             # 构造图
#             g = dgl.graph((from_list, to_list))
#             g.ndata['feats'] = torch.tensor(np.array(node_feats), dtype=torch.float32)
#             # 获取标签
#             label = img_list[0].split('_')[1]
#             if label not in self.label_dict:
#                 self.label_dict[label] = cls
#                 cls += 1
#             graph_label = self.label_dict[label]
#             # 切分训练集和测试集
#             split_type = img_list[0].split('.')[0].split('_')[-1]
#             if split_type == 'train':
#                 self.train_dataset.append((g, graph_label))
#             else:
#                 self.test_dataset.append((g, graph_label))
#         self.n_feats = len(feature)
#
#
# if __name__ == '__main__':
#     loader = ConstructGraphs(data_path='./dataset/subimgs')
#     loader.construct_graphs()
#     np.save('./dataset/fdata/train.npy', loader.train_dataset)
#     np.save('./dataset/fdata/test.npy', loader.test_dataset)


