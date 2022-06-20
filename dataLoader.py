import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from collections import Counter


def dataloader4img(args):
    """
    加载图片数据
    :param args: 模型配置
    :return: train & test
    """
    # different settings
    path = args.path
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    img_size = args.img_size if args is not None else 256
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # get path
    train_folder = os.path.join(path, 'train')
    test_folder = os.path.join(path, 'test')
    # load data
    train_set = datasets.ImageFolder(train_folder, transform=transform_train)
    test_set = datasets.ImageFolder(test_folder, transform=transform_test)
    # report the dataset
    print(train_set.class_to_idx, test_set.class_to_idx)
    print('Train data size:', len(train_set))
    print('Test data size:', len(test_set))
    print('Samples per class in train data:', dict(Counter(train_set.targets)))
    print('Samples per class in test data:', dict(Counter(test_set.targets)))
    return train_set, test_set
