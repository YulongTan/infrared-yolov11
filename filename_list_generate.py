import os
import random

# 输入图片所在目录
image_dir = r"D:\pycharm_proj\yolov11\LLVIP\test\visible"

# 输出文件路径
output_path = "LLVIP/validate.txt"

# 获取所有 .jpg 文件（含扩展名）
filenames = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

# 打乱顺序
random.shuffle(filenames)

# 写入文件
with open(output_path, "w") as f:
    for name in filenames:
        f.write(name + "\n")

print(f"✅ 已生成 {output_path}，共 {len(filenames)} 行，内容为打乱的 JPG 文件名（含扩展名）")
