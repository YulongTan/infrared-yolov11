import os
import torch
from torch.utils.data import DataLoader
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from nets import nn  # 假设nn包含您的YOLO模型定义
from utils.dataset import Dataset  # 假设Dataset是您的数据集加载器

# 初始化量化模块
def initialize_quantization(calib_method="histogram"):
    quant_desc_input = QuantDescriptor(calib_method=calib_method)
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    quant_modules.initialize()

# 替换模型中的模块为量化模块
def replace_layers_with_quantized(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Conv2d):
            quant_module = quant_nn.QuantConv2d(module.in_channels, module.out_channels, 
                                                kernel_size=module.kernel_size, 
                                                stride=module.stride, 
                                                padding=module.padding, 
                                                bias=module.bias is not None)
            setattr(model, name, quant_module)
        elif isinstance(module, torch.nn.Linear):
            quant_module = quant_nn.QuantLinear(module.in_features, module.out_features, 
                                                bias=module.bias is not None)
            setattr(model, name, quant_module)
        else:
            replace_layers_with_quantized(module)

# 准备校准数据加载器
def prepare_calibration_loader(data_dir, input_size, batch_size=1):
    filenames = []
    with open(f'{data_dir}/train2017.txt') as reader:
        for filename in reader.readlines()[:100]:  # 使用前100张图像进行校准
            filenames.append(os.path.join(data_dir, 'images', 'train2017', filename.strip()))
    params = {"input_size": input_size}
    calibration_dataset = Dataset(filenames, input_size, params, augment=False)
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    return calibration_loader

# 校准模型
def calibrate_model(model, calibration_loader):
    # 启用校准器
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
            module.enable_calib()

    # 前向传播以收集统计信息
    model.eval()
    with torch.no_grad():
        for data in calibration_loader:
            images = data[0].float() / 255.0
            model(images)

    # 计算量化范围
    for name, module in model.named_modules():
           if isinstance(module, quant_nn.TensorQuantizer):
                if calibrator == "max":
                    module.load_calib_amax(method="max")
                elif calibrator == "percentile":
                    for p in hist_percentile:
                        module.load_calib_amax(method="percentile", percentile=p)
                elif calibrator in ["mse", "entropy"]:
                    module.load_calib_amax(method=calibrator)
                else:
                    raise ValueError(f"Unsupported calibrator: {calibrator}")
# 保存量化模型
def save_quantized_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"量化后的模型已保存至 {save_path}")

# 加载并替换模型
def load_model(weights_path, num_classes):
    model = torch.load(f='./weights/best.pt', map_location='cpu') # 加载整个检查点文件
    model = model['model'].float().fuse()
    model.eval()
    replace_layers_with_quantized(model)
    return model

# 主函数
def main():
    data_dir = './Dataset/COCO'  # 替换为您的数据目录
    weights_path = './weights/best.pt'  # 替换为您的模型权重路径
    save_path = './weights/quantized_yolo_v11_n.pth'
    input_size = 640
    batch_size = 32
    num_classes = 80

    # 初始化量化模块
    initialize_quantization(calib_method="histogram")

    # 加载模型
    model = load_model(weights_path, num_classes)

    # 准备校准数据
    calibration_loader = prepare_calibration_loader(data_dir, input_size, batch_size)

    # 校准模型
    calibrate_model(model, calibration_loader)

    # 保存量化后的模型
    save_quantized_model(model, save_path)

if __name__ == "__main__":
    main()