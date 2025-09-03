import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType
import numpy as np
import cv2
import os
import random

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset_path, input_size=(640, 640), sample_size=64):
        self.dataset_path = dataset_path
        self.input_size = input_size
        all_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        self.image_paths = random.sample(all_images, min(sample_size, len(all_images)))
        self.index = 0

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        image = cv2.resize(image, self.input_size)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        image_path = self.image_paths[self.index]
        self.index += 1
        return {"input": self.preprocess_image(image_path)}

# 数据集路径和模型路径
# coco dataset path
# dataset_path = "D:\\yolov11\\YOLOv11-pt-master\\Dataset\\COCO\\images\\val2017"
# kitti dataset path
dataset_path = "D:\\yolov11\\YOLOv11-pt-master\\Dataset\\training"
onnx_model_path = 'yolov11_96_kitti.onnx'
quantized_model_path = 'model_static_quant.onnx'

# 创建校准数据读取器
reader = MyCalibrationDataReader(dataset_path)

# 执行静态量化
quantize_static(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QOperator,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,
    extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "CalibMovingAverage": True,
        "CalibMovingAverageConstant": 0.01
    }
)

print("Static quantization completed.")

# 输出验证
reader = MyCalibrationDataReader(dataset_path, sample_size=10)
quantized_model = ort.InferenceSession(quantized_model_path)
float_model = ort.InferenceSession(onnx_model_path)

input_name = quantized_model.get_inputs()[0].name
float_input_name = float_model.get_inputs()[0].name

data = reader.get_next()
while data is not None:
    quant_output = quantized_model.run(None, {input_name: data["input"]})
    float_output = float_model.run(None, {float_input_name: data["input"]})

    mse = np.mean((np.array(float_output[0]) - np.array(quant_output[0])) ** 2)
    print(f"MSE between float and quantized model: {mse}")

    data = reader.get_next()
