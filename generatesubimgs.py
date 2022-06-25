from tqdm import tqdm

from modules.craft.craft import CRAFT
from modules.craft.predictor import *

import numpy as np

import cv2

import os

# STD训练好的模型权重
trained_model = './backbone/craft_mlt_25k.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cuda = False if device == 'cpu' else True
# 图像预处理: 转为tensor + 归一化
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# CRAFT model
craft_model = CRAFT()
print('Loading weights from checkpoint (' + trained_model + ')')
if cuda:
    craft_model.load_state_dict(copyStateDict(torch.load(trained_model)))
else:
    craft_model.load_state_dict(copyStateDict(torch.load(trained_model, map_location='cpu')))
if cuda:
    craft_model = craft_model.cuda()


def generateSubimgs(args_dict, img_path=None, root_path=None, save_path=None):
    """
    :param args_dict: {"net": craft_model, "transform": trans, "device": device}
    :param img_path: 图片路径
    :param root_path: 图片根目录路径
    save_path: 子图保存的路径
    """
    # img name should be f"{idx}_{class}_{train/test}.jpg"
    subimg_dict = {}
    if img_path:
        net, trans, device = [v for k, v in args_dict.items()]
        # get the image name
        num, label, partition = img_path.split('\\')[-1].split('.')[0].split('_')
        # read the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, b = img.shape
        boxes = CRAFT_net(net, trans(img).to(device))  # 返回所有文本框坐标
        if len(boxes) == 0:
            boxes = [[64, 64, 128, 128], [128, 128, 192, 192]]
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            sub_img = img[max(int(ymin)-5, 0):min(int(ymax)+5, img.shape[0]), max(int(xmin)-3, 0):min((int(xmax)+3), img.shape[1]), :]  # 提取包含文本的子图
            # 保存子图
            subimg_name = f"{num}_{label}_{i+1}^{max(int(ymin)-5, 0)},{min(int(ymax)+5, img.shape[0])},{max(int(xmin)-3, 0)},{img.shape[1]}_{partition}"
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                try:
                    cv2.imwrite(os.path.join(save_path, f"{subimg_name}.jpg"),
                                cv2.resize(sub_img, (64, 64), interpolation=cv2.INTER_CUBIC)[:, :, ::-1])
                except:
                    raise Exception("Unvalid path.")
            # log the center point cordinates of the subimg
            subimg_dict[subimg_name] = {'y_m': int((ymin+ymax)/2), 'x_m': int((xmin+xmax)/2),
                                        'h': h, 'w': w}
        return subimg_dict

    # 输入路径时的情况
    elif root_path:
        subimg_center_point_dict = {}
        img_paths = os.listdir(root_path)
        for img_path in tqdm(img_paths):
            img_name = img_path.split('.')[0]
            img_path = os.path.join(root_path, img_path)
            try:
                subimg_dict = generateSubimgs(save_path=save_path, args_dict=args_dict, img_path=img_path)
                subimg_center_point_dict[img_name] = subimg_dict
            except:
                pass
        os.makedirs('./data/dicts', exist_ok=True)
        np.save('./data/dicts/subimg_center_point_dict.npy', subimg_center_point_dict)
    else:
        raise Exception("Neither img_path nor root_path is valid")


if __name__ == '__main__':
    root_path = './data/images'
    save_path = './data/subimgs'
    args_dict = {"net": craft_model, "transform": trans, "device": device}
    generateSubimgs(args_dict=args_dict, root_path=root_path, save_path=save_path)
