import torch
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

import os
from sympy import elliptic_f
import torch
from torch.utils.data import DataLoader
from nets import nn  # 假设nn包含您的YOLO模型定义
from utils.dataset import Dataset  # 假设Dataset是您的数据集加载器
import torch
import torch.nn
import torch.quantization

data_dir = 'D:\\yolov11\\YOLOv11-pt-master\\Dataset\\COCO'
weights_path = './weights/best.pt'  # 替换为您的已训练模型权重路径
def prepare_example_inputs_from_dataset(input_size, num_samples=10, batch_size=1):
    """
    从 Dataset 加载多个样本并生成用于 PTQ 的输入 tuple
    参数：
    - input_size: int, 模型输入尺寸。
    - num_samples: int，加载的样本总数量。
    - batch_size: int，每个批次的样本数量。

    返回：
    - example_inputs: tuple, 包含多个样本批次的 tuple。
    """
    filenames = []
    with open(f'{data_dir}\\train2017.txt') as reader:
        for filename in reader.readlines()[:num_samples]:  # 使用指定数量的图像进行校准
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}\\images\\train2017\\' + filename)

    params = {
        "input_size": input_size,
        # 根据需要设置其他参数，例如图像增强、数据预处理等
    }

    # 初始化 Dataset 实例
    calibration_dataset = Dataset(filenames, input_size, params, augment=False)

    # 从 Dataset 加载数据并将它们组合为一个 tuple
    example_inputs = []
    for i in range(0, len(calibration_dataset), batch_size):
        batch = [calibration_dataset[j][0] for j in range(i, min(i + batch_size, len(calibration_dataset)))]
        batch_tensor = torch.stack(batch)  # 将该批次的样本堆叠为张量
        example_inputs.append(batch_tensor)

    return tuple(example_inputs)  # 返回多个批次组成的 tuple


model = torch.load(f='./weights/best.pt', map_location='cpu') # 加载整个检查点文件
model_fp = model['model'].float().fuse()


#
# post training static quantization
#
example_inputs = prepare_example_inputs_from_dataset(640, num_samples=10, batch_size=1)
model_to_quantize = copy.deepcopy(model_fp)
qconfig_mapping = get_default_qconfig_mapping("qnnpack")
model_to_quantize.eval()
# prepare
model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
# calibrate (not shown)
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)
