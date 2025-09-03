import os
import cv2
import torch
import tqdm
import yaml
import numpy as np
from nets import nn
from utils import util

# 路径配置
root_dir = 'D:/pycharm_proj/yolov11'
data_dir = os.path.join(root_dir, 'LLVIP')
test_img_dir = os.path.join(data_dir, 'test', 'RGBT')
weights_path = os.path.join(root_dir, 'weights', 'best.pt')
yaml_path = os.path.join(root_dir, 'utils', 'kitti.yaml')
video_save_path = os.path.join(root_dir, 'output.mp4')
input_size = 640

def draw_boxes(image, boxes, confs, classes, class_names):
    for box, conf, cls_id in zip(boxes, confs, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls_id)]}: {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

@torch.no_grad()
def main():
    # 加载 class 配置
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    class_names = params['names']

    # 加载模型
    model = torch.load(weights_path, weights_only=False, map_location='cuda')['model'].float().fuse().eval().cuda()

    # 获取测试图像路径并排序
    exts = ('.jpg', '.png', '.jpeg', '.bmp')
    image_paths = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.lower().endswith(exts)]
    image_paths.sort(key=lambda x: os.path.basename(x))  # 按文件名升序排序

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_save_path, fourcc, 10.0, (input_size, input_size))

    for img_path in tqdm.tqdm(image_paths, desc="Processing"):
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[警告] 无法读取图像: {img_path}")
            continue

        image = cv2.resize(image, (input_size, input_size))
        draw_img = image[:, :, :3].copy()  # 仅取 RGB 通道


        sample = image[:, :, [2, 1, 0, 3]]  # 将 BGR-A → RGB-A
        sample = np.transpose(sample, (2, 0, 1))  # HWC → CHW
        img_tensor = torch.from_numpy(sample).unsqueeze(0).cuda().float() / 255.0

        outputs = model(img_tensor)
        outputs = util.non_max_suppression(outputs)[0]

        if outputs is not None and len(outputs):
            boxes = outputs[:, :4].detach().cpu().numpy()
            confs = outputs[:, 4].detach().cpu().numpy()
            classes = outputs[:, 5].detach().cpu().numpy()
            draw_img = draw_boxes(draw_img, boxes, confs, classes, class_names)

        out.write(draw_img)

    out.release()
    print(f"\n✅ 推理完成，视频已保存至：{video_save_path}")

if __name__ == '__main__':
    main()
