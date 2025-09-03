# from PIL import Image
# import matplotlib.pyplot as plt
#
# # 图像路径
# img_path = r"D:/pycharm_proj/yolov11/kaist/kaist_wash_picture_test/lwir/0001.png"
#
# # 加载图像并转换为 RGB（防止原图是 L、LA、RGBA 等模式）
# img_rgb = Image.open(img_path).convert("RGB")
#
# # 显示 RGB 图像
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(img_rgb)
# plt.title("RGB Image")
# plt.axis("off")
#
# # 转换为灰度图
# img_gray = img_rgb.convert("L")
#
# # 显示灰度图
# plt.subplot(1, 2, 2)
# plt.imshow(img_gray, cmap='gray')
# plt.title("Grayscale Image")
# plt.axis("off")
#
# plt.tight_layout()
# plt.show()
# img_gray.save("0001_gray.png")

# from PIL import Image
#
# img = Image.open("D:/pycharm_proj/yolov11/0001_gray.png")
# print(f"图像模式: {img.mode}")
# print(f"通道数: {len(img.getbands())}")

import os
from PIL import Image

# 输入路径
visible_dir = r"D:\pycharm_proj\yolov11\LLVIP\train\visible"
lwir_dir = r"D:\pycharm_proj\yolov11\LLVIP\train\infrared"

# 输出路径
output_dir = r"D:\pycharm_proj\yolov11\LLVIP\train\RGBT"
os.makedirs(output_dir, exist_ok=True)

# 获取 visible 目录下所有 JPG 文件
visible_files = sorted([f for f in os.listdir(visible_dir) if f.lower().endswith(".jpg")])

for filename in visible_files:
    visible_path = os.path.join(visible_dir, filename)
    lwir_path = os.path.join(lwir_dir, filename)

    if not os.path.exists(lwir_path):
        print(f"跳过：找不到匹配的 lwir 图像: {filename}")
        continue

    # 加载图像并转换为 RGB 模式
    visible_img = Image.open(visible_path).convert("RGB")
    lwir_img = Image.open(lwir_path).convert("RGB")

    # 直接取 LWIR 图像的任意通道（因为 R=G=B）
    alpha = lwir_img.getchannel("R")

    # 合成 RGBA 图像
    rgba_img = Image.merge("RGBA", (*visible_img.split(), alpha))

    # 修改输出文件名为 .png
    output_filename = os.path.splitext(filename)[0] + ".png"
    output_path = os.path.join(output_dir, output_filename)

    # 保存为 PNG（支持 RGBA）
    rgba_img.save(output_path)

    print(f"已保存：{output_path}")

print("✅ 全部处理完成。")


# from PIL import Image
# import matplotlib.pyplot as plt
#
# # 图像路径
# img_path = r"D:/pycharm_proj/yolov11/kaist/kaist_wash_picture_train/0001.png"
#
# # 打开图像并确保是 RGBA 模式
# img = Image.open(img_path).convert("RGBA")
#
# # 拆分通道：R, G, B, A
# r, g, b, a = img.split()
#
# # 合成 RGB 图像（不带透明度）
# rgb_img = Image.merge("RGB", (r, g, b))
#
# # 显示图像
# plt.figure(figsize=(10, 4))
#
# # RGB 图像
# plt.subplot(1, 2, 1)
# plt.imshow(rgb_img)
# plt.title("RGB Image")
# plt.axis("off")
#
# # Alpha 通道（灰度显示）
# plt.subplot(1, 2, 2)
# plt.imshow(a, cmap='gray')
# plt.title("Alpha Channel (Grayscale)")
# plt.axis("off")
#
# plt.tight_layout()
# plt.show()
