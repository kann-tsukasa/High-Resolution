import pandas as pd

import os
import torch

import torchvision.transforms as transforms

import cv2
import numpy as np
from collections import OrderedDict

from modules.craft import utils

from sklearn.metrics import confusion_matrix

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_confusion_matrix(true_label, pred_label, categories):
    mat = confusion_matrix(true_label, pred_label, labels=categories)
    df = pd.DataFrame(mat)
    df.columns = categories
    df.index = [x + "_true" for x in categories]
    return df


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def detect(image, model, crop_size=224, device="cpu"):
    device = torch.device(device)
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(device)
    output = model(image)[0]
    predicted = torch.max(output.data, 1)[1].data
    return predicted.cpu().numpy()[0]

def computebox(box):
    """
    计算文本框的xmin, ymin, xmax, ymax

    """
    a = box.astype(np.int32)
    xmax, ymax = a.max(0)
    xmin, ymin = a.min(0)
    width = xmax - xmin
    height = ymax - ymin
    return max(xmin, 0), max(ymin, 0), int(xmax), int(ymax)


def CRAFT_net(net, image):
    """
    返回图像image所有的文本区域坐标: (xmin,ymin,xmax, ymax)

    """
    text_threshold = 0.7
    link_threshold = 0.4
    low_text = 0.4
    poly = False  # 输出boxes即可
    ratio = 1  # 不对图像进行缩放

    net.eval()
    with torch.no_grad():
        y, feature = net(image.unsqueeze(0))

    ratio_w = ratio_h = 1 / ratio
    # 生存 score 和 link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    boxes, _ = utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # 调整框的坐标
    boxes = utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    res = []
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = computebox(box)
        res.append(np.array([xmin - 2, ymin - 2, xmax + 2, ymax + 2]).astype(int))

    return res


def getBoxes(y_pred,
             detection_threshold=0.7,
             text_threshold=0.4,
             link_threshold=0.4,
             size_threshold=10):
    box_groups = []
    for y_pred_cur in y_pred:
        # Prepare data
        textmap = y_pred_cur[..., 0].copy()
        linkmap = y_pred_cur[..., 1].copy()
        img_h, img_w = textmap.shape

        _, text_score = cv2.threshold(textmap,
                                      thresh=text_threshold,
                                      maxval=1,
                                      type=cv2.THRESH_BINARY)
        _, link_score = cv2.threshold(linkmap,
                                      thresh=link_threshold,
                                      maxval=1,
                                      type=cv2.THRESH_BINARY)
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(np.clip(
            text_score + link_score, 0, 1).astype('uint8'),
                                                                          connectivity=4)
        boxes = []
        for component_id in range(1, n_components):
            # Filter by size
            size = stats[component_id, cv2.CC_STAT_AREA]

            if size < size_threshold:
                continue

            # If the maximum value within this connected component is less than
            # text threshold, we skip it.
            if np.max(textmap[labels == component_id]) < detection_threshold:
                continue

            # Make segmentation map. It is 255 where we find text, 0 otherwise.
            segmap = np.zeros_like(textmap)
            segmap[labels == component_id] = 255
            segmap[np.logical_and(link_score, text_score)] = 0
            x, y, w, h = [
                stats[component_id, key] for key in
                [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
            ]

            # Expand the elements of the segmentation map
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, sy = max(x - niter, 0), max(y - niter, 0)
            ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
            segmap[sy:ey, sx:ex] = cv2.dilate(
                segmap[sy:ey, sx:ex],
                cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)))

            # Make rotated box from contour
            contours = cv2.findContours(segmap.astype('uint8'),
                                        mode=cv2.RETR_TREE,
                                        method=cv2.CHAIN_APPROX_SIMPLE)[-2]
            contour = contours[0]
            box = cv2.boxPoints(cv2.minAreaRect(contour))

            # Check to see if we have a diamond
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = contour[:, 0, 0].min(), contour[:, 0, 0].max()
                t, b = contour[:, 0, 1].min(), contour[:, 0, 1].max()
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
            else:
                # Make clock-wise order
                box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
            boxes.append(2 * box)
        box_groups.append(np.array(boxes))
    return box_groups
