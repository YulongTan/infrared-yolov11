import os

# 原始路径
folder = r"D:\pycharm_proj\yolov11\kaist\kaist_wash_annotation_test"

# 获取所有 XML 文件，按文件名排序（也可按修改时间、创建时间排序）
xml_files = sorted([f for f in os.listdir(folder) if f.endswith(".xml")])

# 确保文件数量不超过 9999
if len(xml_files) > 9999:
    raise ValueError("文件数超过4位命名限制")

# 开始重命名
for idx, old_name in enumerate(xml_files, start=1):
    new_name = f"{idx:04d}.xml"  # 格式如 0001.xml
    old_path = os.path.join(folder, old_name)
    new_path = os.path.join(folder, new_name)

    # 避免覆盖已有目标文件名（若源文件与目标名冲突）
    if os.path.exists(new_path):
        raise FileExistsError(f"目标文件 {new_name} 已存在，重命名可能冲突")

    os.rename(old_path, new_path)

print(f"共重命名 {len(xml_files)} 个 XML 文件，起始名为 0001.xml")
