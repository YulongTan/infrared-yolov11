import glob
import os

def split_video_labels(video_txt_file, output_dir):
    """
    将一个视频的标签转换为对应每一帧的单独 txt 文件。
    
    video_txt_file: 输入的视频标签文件（.txt格式）。
    output_dir: 输出目录，用于存放每帧对应的标签文件。
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取视频标签文件
    try:
        with open(video_txt_file, 'r') as file:
            frame_data = {}
            for line in file:
                label_data = line.strip().split(' ')  # 分割每一行的标签数据
                frame_id = int(label_data[0])  # 第一列是图像帧编号
                category = label_data[1]  # 第二列是类别
                
                # 将标签存储在对应帧编号的字典里
                if frame_id not in frame_data:
                    frame_data[frame_id] = []
                frame_data[frame_id].append(label_data[1:])
                
    except IOError as e:
        print(f"读取文件错误: {e}")
        return
    
    # 为每一帧创建单独的 txt 文件
    for frame_id, annotations in frame_data.items():
        txt_filename = os.path.join(output_dir, f"{frame_id:06d}.txt")  # 帧编号作为文件名
        try:
            with open(txt_filename, 'w') as frame_file:
                for annotation in annotations:
                    frame_file.write(" ".join(annotation) + '\n')  # 写入每个标签信息
        except IOError as e:
            print(f"写入文件错误: {e}")

def merge(line):
    """
    合并标签数据到一行字符串。
    """
    each_line = ''
    for i in range(len(line)):
        if i != len(line) - 1:
            each_line = each_line + line[i] + ' '
        else:
            each_line = each_line + line[i]
    each_line = each_line + '\n'
    return each_line

def process_labels(txt_list):
    """
    处理标签，合并类别并忽略不需要的类别。
    """
    print('Before modify categories are:\n')
    show_category(txt_list)

    for item in txt_list:
        new_txt = []
        try:
            with open(item, 'r') as r_tdf:
                for each_line in r_tdf:
                    labeldata = each_line.strip().split(' ')
                    if labeldata[1] in ['Truck', 'Van', 'Tram', 'Misc']:  # 合并汽车类
                        labeldata[1] = 'Car'
                    if labeldata[1] == 'Person_sitting':  # 合并行人类
                        labeldata[1] = 'Pedestrian'
                    if labeldata[1] == 'DontCare':  # 忽略Dontcare类
                        continue
                    new_txt.append(merge(labeldata))  # 重新写入新的txt文件
                    
            with open(item, 'w+') as w_tdf:  # w+是打开原文件将内容删除，另写新内容进去
                for temp in new_txt:
                    w_tdf.write(temp)
        except IOError as ioerr:
            print(f'文件错误: {ioerr}')
    
    print('\nAfter modify categories are:\n')
    show_category(txt_list)

def show_category(txt_list):
    """
    输出所有类别。
    """
    category_list = []
    for item in txt_list:
        try:
            with open(item) as tdf:
                for each_line in tdf:
                    labeldata = each_line.strip().split(' ')  # 去掉前后多余的字符并把其分开
                    category_list.append(labeldata[0])  # 只要第一个字段，即类别
        except IOError as ioerr:
            print(f'文件错误: {ioerr}')
    print(set(category_list))  # 输出集合


# 示例用法
video_txt_file = "D:\\yolov11\\YOLOv11-pt-master\\Dataset\\video\\tracking\\training\\label_01\\0004.txt"  # 视频的标签文件路径
output_dir = "D:\\yolov11\\YOLOv11-pt-master\\Dataset\\video\\tracking\\training\\label"   # 输出文件夹路径
split_video_labels(video_txt_file, output_dir)  # 分割视频标签为每一帧的标签文件

# 获取所有图片标签文件列表
txt_list = glob.glob(os.path.join(output_dir, "*.txt"))  # 获取文件夹下所有 txt 文件
process_labels(txt_list)  # 处理每个帧的标签，合并类别
