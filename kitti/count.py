import os

def count_classes(label_dir, file_list):
    """
    统计指定文件列表中每个类别的总数。

    Args:
        label_dir (str): 标签文件的根目录。
        file_list (str): 包含图片地址的文件（train.txt 或 val.txt）。

    Returns:
        dict: 每个类别的计数结果。
    """
    # 初始化类别计数
    class_counts = {0: 0, 1: 0, 2: 0}

    # 读取文件列表
    with open(file_list, 'r') as f:
        lines = f.readlines()

    # 遍历每个文件路径
    for line in lines:
        image_path = line.strip()
        label_file = os.path.join(label_dir, os.path.basename(image_path).replace('.png', '.txt'))

        # 检查标签文件是否存在
        if os.path.exists(label_file):
            with open(label_file, 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])  # 读取类别ID
                    if class_id in class_counts:
                        class_counts[class_id] += 1

    return class_counts

if __name__ == "__main__":
    # 设置标签文件目录和列表文件路径
    label_dir = "D:\\yolov11\\YOLOv11-pt-master\\Dataset\\training\\labels"  # 修改为实际的标签文件目录路径
    train_file = "D:\\yolov11\\YOLOv11-pt-master\\Dataset\\training\\train.txt"  # 训练集文件列表
    val_file = "D:\\yolov11\\YOLOv11-pt-master\\Dataset\\training\\val.txt"  # 测试集文件列表

    # 统计训练集类别数量
    if os.path.exists(train_file):
        train_class_counts = count_classes(label_dir, train_file)
        print("训练集类别统计结果:")
        print(f"Car (0): {train_class_counts[0]}")
        print(f"Pedestrian (1): {train_class_counts[1]}")
        print(f"Cyclist (2): {train_class_counts[2]}")
    else:
        print(f"训练集文件 {train_file} 不存在！")

    # 统计测试集类别数量
    if os.path.exists(val_file):
        val_class_counts = count_classes(label_dir, val_file)
        print("测试集类别统计结果:")
        print(f"Car (0): {val_class_counts[0]}")
        print(f"Pedestrian (1): {val_class_counts[1]}")
        print(f"Cyclist (2): {val_class_counts[2]}")
    else:
        print(f"测试集文件 {val_file} 不存在！")
