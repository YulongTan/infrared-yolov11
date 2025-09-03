import os
import cv2
import onnxruntime as ort
import numpy as np
import torch
from utils import util
import random
import time


# 加载 ONNX 模型
def load_onnx_model(onnx_path):
    session = ort.InferenceSession(onnx_path)
    return session


# 获取随机数据增强的函数
def resample():
    choices = (cv2.INTER_AREA,
               cv2.INTER_CUBIC,
               cv2.INTER_LINEAR,
               cv2.INTER_NEAREST,
               cv2.INTER_LANCZOS4)
    return random.choice(seq=choices)


# Resize 和填充图片，使其满足输入尺寸的要求
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


# 图片预处理
def preprocess_frame(frame, input_size=640, augment=False):
    orig_h, orig_w, _ = frame.shape
    image, ratio, pad = resize(frame, input_size, augment)

    return image, (orig_w, orig_h), ratio, pad


# 使用 ONNX 模型进行推理
def infer_onnx(session, frame, input_size=640, augment=False):
    # 预处理
    img, (orig_w, orig_h), ratio, pad = preprocess_frame(frame, input_size, augment)
    
    # 将图像从 BGR 转换为 RGB，归一化到 [0, 1]
    img_resized = img.astype(np.float32) / 255.0
    img_resized = img_resized.transpose((2,0,1))[::-1] # 转换为 NCHW 格式
    img_resized = np.expand_dims(img_resized, axis=0)  # 添加 batch 维度
    # 获取模型输入名称
    input_name = session.get_inputs()[0].name

    # 执行推理
    outputs = session.run(None, {input_name: img_resized})
    
    return outputs, (orig_w, orig_h), ratio, pad, img_resized


def process_images(onnx_model_path, image_folder, output_folder, input_size=640, augment=False):
    """
    处理图片流并绘制检测结果。
    :param onnx_model_path: ONNX 模型路径。
    :param image_folder: 图片存储目录路径。
    :param output_folder: 绘制结果保存目录路径。
    :param input_size: 模型输入尺寸。
    :param augment: 是否启用数据增强。
    """
    # 加载 ONNX 模型
    session = load_onnx_model(onnx_model_path)

    # 获取目录下的所有图片文件，按文件名排序
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing {image_path}...")

        # 读取图像
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_path}")
            continue

        # 开始计时
        start_time = time.time()

        # 推理
        outputs, (orig_w, orig_h), ratio, pad, img_resized = infer_onnx(session, frame, input_size, augment)
        outputs = torch.tensor(outputs[0])  # ONNX 输出转换为 PyTorch 张量

        # 使用 NMS 过滤结果
        detections = util.non_max_suppression(outputs)
        # 在图像上绘制检测框
        for det in detections:
            if det is None or len(det) == 0:
                continue

            # 还原检测框的尺寸到原始图像比例
            for *xyxy, conf, cls in det:
                # 获取框的坐标 (xyxy) 为左上和右下角坐标
                x_min, y_min, x_max, y_max = xyxy
                
                # 将框的坐标还原到原始图像尺寸
                x_min = int(((x_min - pad[0]) /(input_size - 2 * pad[0])) * orig_w)
                y_min = int(((y_min - pad[1]) /(input_size - 2 * pad[1])) * orig_h)
                x_max = int(((x_max - pad[0]) /(input_size - 2 * pad[0])) * orig_w)
                y_max = int(((y_max - pad[1]) /(input_size - 2 * pad[1])) * orig_h)

                label = f'{int(cls)} {conf:.2f}'

                # 绘制框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # 绘制标签
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 计算处理时间
        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

        # 保存绘制结果
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, frame)
        print(f"Saved result to {output_path}")


if __name__ == "__main__":
    # 参数设置
    onnx_model_path = 'model_static_quant_normalized_q8_entropy_2false.onnx'  # 修改为你的 ONNX 模型路径
    image_folder = "/data/wwh/dataset/yolo/0004"  # 图片目录路径
    output_folder = "/data/wwh/dataset/yolo/0004_output"  # 结果保存目录

    # 运行图片流处理
    process_images(onnx_model_path, image_folder, output_folder)
