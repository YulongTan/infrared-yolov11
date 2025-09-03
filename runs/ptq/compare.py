import onnxruntime as ort
import numpy as np
import torch
import cv2
import random
from utils.dataset import Dataset  # 你的数据集定义
from nets import nn  # 你的模型定义
import os

# 数据路径
dataset_path = "D:/yolov11/YOLOv11-pt-master/Dataset/COCO/images/val2017"
onnx_model_path = "yolov11_n.onnx"  # 原始浮点模型
quantized_model_path = "model_static_quant.onnx"  # 量化后的模型

# 随机选一张图片
def get_random_image(dataset_path):
    all_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    image_path = random.choice(all_images)
    return image_path

# 图像预处理（与训练时一致）
def preprocess_image(image_path, input_size=(640, 640)):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
    image = cv2.resize(image, input_size)
    image = image.astype(np.float32) / 255.0  # 归一化到 [0, 1]
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # 添加批次维度
    return image

# 比较量化前后的输出
def compare_models(image_path, onnx_model_path, quantized_model_path):
    input_data = preprocess_image(image_path)

    # 加载浮点模型
    float_model = ort.InferenceSession(onnx_model_path)
    float_input_name = float_model.get_inputs()[0].name
    float_output = float_model.run(None, {float_input_name: input_data})[0]

    # 加载量化模型
    quantized_model = ort.InferenceSession(quantized_model_path)
    quant_input_name = quantized_model.get_inputs()[0].name
    quantized_output = quantized_model.run(None, {quant_input_name: input_data})[0]

    # 比较结果
    float_output_np = np.array(float_output)
    quantized_output_np = np.array(quantized_output)

    mse = np.mean((float_output_np - quantized_output_np) ** 2)  # 均方误差
    cosine_similarity = np.dot(float_output_np.flatten(), quantized_output_np.flatten()) / (
        np.linalg.norm(float_output_np) * np.linalg.norm(quantized_output_np)
    )  # 余弦相似度

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Cosine Similarity: {cosine_similarity}")
    return float_output_np, quantized_output_np

# 主函数
if __name__ == "__main__":
    # 随机选一张图片
    random_image_path = get_random_image(dataset_path)
    print(f"Randomly selected image: {random_image_path}")

    # 比较模型输出
    float_output, quantized_output = compare_models(random_image_path, onnx_model_path, quantized_model_path)

    # 打印结果
    print("Float Model Output (first 5 elements):", float_output.flatten()[:5])
    print("Quantized Model Output (first 5 elements):", quantized_output.flatten()[:5])
