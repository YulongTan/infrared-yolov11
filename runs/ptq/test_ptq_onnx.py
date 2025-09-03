import os
import torch
import tqdm
import onnxruntime as ort
import numpy as np
from argparse import ArgumentParser
from utils import util
from utils.dataset import Dataset
import yaml

data_dir = 'D:\\yolov11\\YOLOv11-pt-master\\Dataset\\training'


@torch.no_grad()
def test_onnx_model(args, params, onnx_model_path):
    """
    测试量化后的 ONNX 模型。
    Args:
        args: 命令行参数。
        params: 配置参数。
        onnx_model_path: ONNX 模型路径。
    """
    # 加载测试数据集
    filenames = []
    with open(f'{data_dir}\\val.txt') as reader:
        for filename in reader.readlines():
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}\\images\\' + filename)

    dataset = Dataset(filenames, args.input_size, params, augment=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn
    )

    # 加载 ONNX 模型
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name  # 模型输入名称
    output_name = session.get_outputs()[0].name  # 模型输出名称

    # 初始化评估指标
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cpu()
    n_iou = iou_v.numel()

    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        # 预处理输入
        samples = samples.numpy().astype(np.float32) / 255.0  # 归一化到 [0, 1]
        # 推理
        outputs = session.run([output_name], {input_name: samples})[0]
        outputs = torch.tensor(outputs)  # 转为 PyTorch 张量以便后续处理
        # NMS
        breakpoint()
        outputs = util.non_max_suppression(outputs)

        # 计算指标
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx]
            box = targets['box'][idx]

            cls = cls.cpu()
            box = box.cpu()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cpu()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cpu(), cls.squeeze(-1)))
                continue
            # Evaluate
            if cls.shape[0]:
                scale = torch.tensor((samples.shape[-1], samples.shape[-2], samples.shape[-1], samples.shape[-2])).cpu()
                target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                metric = util.compute_metric(output[:, :6], target, iou_v)
            # Append
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    # 计算最终结果
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics)
        print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    else:
        print("No valid predictions.")

    return mean_ap, map50, m_rec, m_pre


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int, help='模型输入大小')
    parser.add_argument('--onnx-path', default='yolov11_96_kitti.onnx', type=str, help='ONNX 模型路径')
    # parser.add_argument('--onnx-path', default='yolov11_n.onnx', type=str, help='ONNX 模型路径')
    args = parser.parse_args()

    # 加载配置参数
    with open('utils/kitti.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    # 测试量化后的 ONNX 模型
    mean_ap, map50, m_rec, m_pre = test_onnx_model(args, params, args.onnx_path)

    # 输出测试结果
    print(f"Results: mAP: {mean_ap}, mAP50: {map50}, Recall: {m_rec}, Precision: {m_pre}")


if __name__ == "__main__":
    main()
