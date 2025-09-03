import os
from sympy import elliptic_f
import torch
from torch.utils.data import DataLoader
from nets import nn  # 假设nn包含您的YOLO模型定义
from utils.dataset import Dataset  # 假设Dataset是您的数据集加载器
import torch
import torch.nn
import torch.quantization

import yaml
# 设置数据目录和模型路径
data_dir = 'D:\\yolov11\\YOLOv11-pt-master\\Dataset\\COCO'
weights_path = './weights/best.pt'  # 替换为您的已训练模型权重路径

# 设置模型参数
num_classes = 80  # 设置类别数量
input_size = 640  # 输入图像尺寸
batch_size = 32  # 校准批次大小
class QuantizedConvLayer(torch.nn.Module):
    def __init__(self, conv_layer):
        super(QuantizedConvLayer, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = conv_layer  # 直接使用已有的 Conv2d 层
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        return x
    
class QuantizedBmmLayer(torch.nn.Module):
    def __init__(self):
        super(QuantizedBmmLayer, self).__init__()
        self.dequant = torch.quantization.DeQuantStub()
        self.quant = torch.quantization.QuantStub()

    def forward(self, x, y):
        x = self.dequant(x)
        y = self.dequant(y)
        output = torch.bmm(x, y)
        return self.quant(output)

class QuantizedActivationLayer(torch.nn.Module):
    def __init__(self):
        super(QuantizedActivationLayer, self).__init__()
        self.dequant = torch.quantization.DeQuantStub()
        self.silu = torch.nn.SiLU()  # 如果需要替换为其他激活函数，请在这里更改
    
    def forward(self, x):
        x = self.dequant(x) 
        x = self.silu(x)
        return x
    
# def set_attention_qconfig_none(model):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Attention):
#             module.qconfig = None  # 禁用 Attention 模块的量化

# 将模型的各个模块替换为新的量化模块
def replace_layers_with_quantized(model):
    for name, module in model.named_children():
        # 判断是否是 Attention 模块，如果是则跳过量化
        # if isinstance(module, nn.Attention):
        #     continue  # 跳过量化，不做任何替换

        if isinstance(module, torch.nn.Conv2d):
            # 替换 Conv2d 层为量化卷积层
            setattr(model, name, QuantizedConvLayer(module))

        elif isinstance(module, torch.nn.SiLU):  # 如果需要替换其他激活函数
            setattr(model, name, QuantizedActivationLayer())

        elif isinstance(module, torch.nn.Bilinear):  # 如果需要替换其他层
            setattr(model, name, QuantizedBmmLayer())

        else:
            replace_layers_with_quantized(module)  # 递归替换子模块


# def load_model(weights_path, num_classes):
#     """
#     加载训练好的YOLO模型权重
#     """
#     with open('utils/args.yaml', errors='ignore') as f:
#         params = yaml.safe_load(f)
#     model = nn.yolo_v11_n(num_classes)
#     model.load_state_dict(torch.load(weights_path, map_location='cpu'))
#     model.eval()
#     return model

def prepare_calibration_loader(input_size, batch_size=1):
    """
    准备校准数据加载器，只需要少量数据
    """
    filenames = []
    with open(f'{data_dir}\\train2017.txt') as reader:
        for filename in reader.readlines()[:1]:  # 使用前100张图像进行校准
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}\\images\\train2017\\' + filename)

    params = {
        "input_size": input_size,
        # 根据需要设置其他参数，例如图像增强、数据预处理等
    }
    calibration_dataset = Dataset(filenames, input_size, params, augment=False)
    calibration_loader = DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False)
    return calibration_loader

def ptq_calibration(model, calibration_loader):
    """
    使用校准数据对模型进行量化校准
    """

    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')  # 使用8-bit量化
    model_fp32_prepared = torch.quantization.prepare(model)
    
    # 使用校准数据进行前向传播，收集统计信息
    with torch.no_grad():
        for data in calibration_loader:
            images = data[0].float() /255.0 # 只获取图像部分

            # 如果图像部分在第一个元素，可以这样处理
            model_fp32_prepared(images)
    
    # 转换为量化模型
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    torch.save(model_int8.state_dict(), 'weights/quantized_yolo_v11_n.pt')
    breakpoint()

    # 验证量化模型

    return model

def main():
    # 加载已训练模型



    model = torch.load(f='./weights/best.pt', map_location='cpu') # 加载整个检查点文件
    model = model['model'].float().fuse()
    model.eval()
    replace_layers_with_quantized(model)

    # 准备校准数据
    calibration_loader = prepare_calibration_loader(input_size=input_size, batch_size=batch_size)

    # 对模型进行PTQ量化
    quantized_model = ptq_calibration(model, calibration_loader)

    # 保存量化模型
    torch.save(quantized_model.state_dict(), 'weights/quantized_yolo_v11_n.pt')
    print("量化后的模型已保存至 'weights/quantized_yolo_v11_n.pt'")

if __name__ == "__main__":
    main()