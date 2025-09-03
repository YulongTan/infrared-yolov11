import os
import numpy as np
import torch
from glob import glob
from utils.util import compute_ap  # 假设你有一个util模块，包含用于计算IoU和AP等的函数
import utils.util as util
import cv2

def resize(image, input_size, augment=False):
    h, w = image.shape[:2]
    r = min(input_size / h, input_size / w)

    if not augment:
        r = min(r, 1.0)

    new_w = int(w * r)
    new_h = int(h * r)

    image_resized = cv2.resize(image, (new_w, new_h), interpolation=resample() if augment else cv2.INTER_LINEAR)

    # 填充
    pad_w = (input_size - new_w) / 2
    pad_h = (input_size - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # 确保填充后的图像大小是 input_size
    image_padded = cv2.resize(image_padded, (input_size, input_size))

    return image_padded, (r, r), (pad_w, pad_h)

def load_outputs(npy_dir):
    """
    从npy文件夹加载每张图片的NMS后的预测结果。
    npy_dir: 存储npy文件的目录
    返回：一个列表，每个元素是一个包含该图片预测的np.array
    """
    outputs = []
    # # 获取npy文件夹中的所有文件并按文件名排序
    # npy_files = sorted(os.listdir(npy_dir))
    # 按照npy文件名最后一个数字进行排序
    npy_files = sorted(os.listdir(npy_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for npy_file in npy_files:
        if npy_file.endswith('.npy'):  # 确保只读取.npy文件
            npy_file_path = os.path.join(npy_dir, npy_file)  # 获取完整文件路径
            output = np.load(npy_file_path)  # 读取npy文件
            outputs.append(output)
            
    return outputs

def load_targets(txt_dir, img_dir, input_size):
    """
    从txt文件夹加载每张图片的真实标签，并考虑padding。
    txt_dir: 存储txt文件的目录
    img_dir: 存储图像的目录
    input_size: 输入的图像尺寸（宽和高相同）
    返回：一个列表，每个元素是一个包含该图片真实标签的字典（索引、类别、框坐标）
    """
    targets = []

    for txt_file in sorted(glob(os.path.join(txt_dir, "*.txt"))):
        # 获取对应的图像路径
        img_path = os.path.join(img_dir, os.path.basename(txt_file).replace('.txt', '.png'))
        image = cv2.imread(img_path)  # 读取图像
        h, w = image.shape[:2]  # 获取图像原始尺寸

        # 调用resize函数来获取padding信息
        image_padded, scale, pad = resize(image, input_size)
        
        cls = []
        box = []
      
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split()
                cls.append(int(data[0]))  # 类别
                box.append([float(coord) for coord in data[1:]])  # 边界框

        # 将边界框进行缩放和padding调整
        cls = np.array(cls)
        box = np.array(box)
        idx = np.zeros(len(cls))  # 初始化idx为全0，长度与cls相同
        # 调整边界框：根据缩放和padding的比例
        box[:, [0]] = (box[:, [0]] * w + pad[0]) / max(w, input_size)  # x1
        box[:, [2]] = (box[:, [2]] * w ) / max(w, input_size)
        box[:, [1]] = (box[:, [1]] * h * input_size / max(w, h, input_size)+ pad[1])/ max(h, input_size)
        box[:, [3]] = (box[:, [3]] * h  * input_size/ max(w, h, input_size)) / max(h, input_size)

        targets.append({'cls': cls, 'box': box, 'idx': idx})
    return targets


def evaluate_map(npy_dir, txt_dir, img_dir, iou_v=0.5, n_iou=10):
    """
    根据npy文件和txt文件计算mAP
    npy_dir: 存储npy预测结果的目录
    txt_dir: 存储txt标签的目录
    iou_v: IOU阈值
    n_iou: IOU阈值数量
    """
    # 加载NMS后的预测结果和真实标签
    outputs = load_outputs(npy_dir)
    targets = load_targets(txt_dir, img_dir, input_size=640)
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cpu()
    # output 坐标归一化
    # for output in outputs:
    #     output[:, :4] = output[:, :4]
    # 存储所有的metrics
    metrics = []
    for i, (output, target) in enumerate(zip(outputs, targets)):
        # target['idx']是每个图片的目标索引
        cls = target['cls']  # 所有目标的类别
        box = target['box']  # 所有目标的边界框
        
        # 将cls和box转换为PyTorch张量
        cls = torch.tensor(cls).cpu()  # 类别
        box = torch.tensor(box).cpu()  # 边界框
        
        cls = cls.unsqueeze(-1)  # 为cls添加一个额外的维度

        # 如果没有检测到任何目标，直接跳过
        if output.shape[0] == 0:  # 没有预测结果
            if cls.shape[0]:  # 如果存在标签
                # 添加一个全为False的metric
                metrics.append((torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cpu(),
                                *torch.zeros((2, 0)).cpu(), cls.squeeze(-1)))
            continue
        # 计算metrics
        metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cpu()
        if cls.shape[0]:
            # 处理坐标和类别
            scale = 640
           
            target_tensor = torch.cat(tensors = (cls, util.wh2xy(box) * scale), dim=1)
            breakpoint()
            metric = util.compute_metric(torch.tensor(output[:, :6]), target_tensor, iou_v)
        # 存储每张图片的结果
        metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))
    # 计算最终结果
    metrics = [torch.cat([torch.tensor(m) for m in x], dim=0).cpu().numpy() for x in zip(*metrics)]
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)
        print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    else:
        print("No valid predictions.")

    return mean_ap, map50, m_rec, m_pre


# 调用evaluate_map函数
npy_dir = "./yolo_sift/npy"  # NMS后预测结果的文件夹路径
txt_dir = "./Dataset/video/tracking/training/label_process"  # 真实标签文件夹路径
img_dir = "./Dataset/video/tracking/training/image_02/0004"  # 图像文件夹路径
mean_ap, map50, m_rec, m_pre = evaluate_map(npy_dir, txt_dir, img_dir)
