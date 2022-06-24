#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File     :    generate_cnn_dataset.py
@Time     :    2022/5/23 12:20
@Author   :    Siyi Han
@Version  :    1.0

'''


import cv2
import os
from tqdm import tqdm


if __name__ == '__main__':
    images_path = "./images"
    save_path = "./main"
    train_path = os.path.join(save_path, "train")
    test_path = os.path.join(save_path, "test")
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    files = os.listdir(images_path)
    for file in tqdm(files):
        if file == ".DS_Store":
            continue
        try:
            _, cls, split = file.split("_")
        except:
            continue
        if split == "train.jpg":
            img = cv2.imread(os.path.join(images_path, file))
            cls_path = os.path.join(train_path, cls)
            os.makedirs(cls_path, exist_ok=True)
            cv2.imwrite(os.path.join(cls_path, file), img)
        else:
            img = cv2.imread(os.path.join(images_path, file))
            cls_path = os.path.join(test_path, cls)
            os.makedirs(cls_path, exist_ok=True)
            cv2.imwrite(os.path.join(cls_path, file), img)

