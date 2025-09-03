# import os
# import xml.etree.ElementTree as ET
#
# # 标签类别映射
# label_map = {
#     "person": 0
# }
#
# # 输入和输出路径
# xml_dir = r"D:\pycharm_proj\yolov11\LLVIP\Annotations"
#
# # 遍历目录下所有 XML 文件
# for filename in os.listdir(xml_dir):
#     if not filename.endswith(".xml"):
#         continue
#
#     xml_path = os.path.join(xml_dir, filename)
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#
#     size = root.find("size")
#     img_w = float(size.find("width").text)
#     img_h = float(size.find("height").text)
#
#     # 输出 .txt 文件路径（同名）
#     txt_filename = os.path.splitext(filename)[0] + ".txt"
#     txt_path = os.path.join(xml_dir, txt_filename)
#
#     lines = []
#
#     for obj in root.findall("object"):
#         name = obj.find("name").text.strip()
#
#         # 忽略无效类别
#         if name not in label_map:
#             continue
#
#         class_id = label_map[name]
#
#         bndbox = obj.find("bndbox")
#         xmin = float(bndbox.find("xmin").text)
#         ymin = float(bndbox.find("ymin").text)
#         xmax = float(bndbox.find("xmax").text)
#         ymax = float(bndbox.find("ymax").text)
#
#         # 计算中心点、宽高，并归一化
#         x_center = ((xmin + xmax) / 2) / img_w
#         y_center = ((ymin + ymax) / 2) / img_h
#         width = (xmax - xmin) / img_w
#         height = (ymax - ymin) / img_h
#
#         # 格式化为 YOLO 标签行
#         line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
#         lines.append(line)
#
#     # 写入 txt 文件
#     with open(txt_path, "w") as f:
#         f.write("\n".join(lines))
#
#     print(f"✅ 处理完成：{txt_filename}")
#
# print("🎉 所有 XML 已转换完成。")

#从总标签文件夹中找出trian和test对应的标签，另存
import os
import shutil

# 路径配置
jpg_dir = r"D:\pycharm_proj\yolov11\LLVIP\test\visible"
label_src_dir = r"D:\pycharm_proj\yolov11\LLVIP\labels"
label_dst_dir = r"D:\pycharm_proj\yolov11\LLVIP\test\labels"

# 确保目标目录存在
os.makedirs(label_dst_dir, exist_ok=True)

# 遍历visible目录下的jpg文件
for filename in os.listdir(jpg_dir):
    if filename.lower().endswith('.jpg'):
        basename = os.path.splitext(filename)[0]
        label_file = f"{basename}.txt"
        src_label_path = os.path.join(label_src_dir, label_file)
        dst_label_path = os.path.join(label_dst_dir, label_file)

        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dst_label_path)
        else:
            print(f"[警告] 找不到标签文件：{src_label_path}")

print("复制完成。")
