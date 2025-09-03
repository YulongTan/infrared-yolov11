import glob
import random

def splitdataset():
    random.seed(1234)
    
    label_path = 'D:/yolov11/YOLOv11-pt-master/Dataset/training/labels/*.txt'       # 这里修改为上一步保存的新标签的位置
    filename_list = glob.glob(label_path)
    num_file = len(filename_list)
    
    val = 0.2      # 验证集的比例
    
    try:
        val_file = open('D:/yolov11/YOLOv11-pt-master/Dataset/training/val.txt', 'w', encoding='utf-8')    # 包含验证集的txt文件，修改为自己想要保存的位置
        train_file = open('D:/yolov11/YOLOv11-pt-master/Dataset/training/train.txt', 'w', encoding='utf-8')  # 包含训练集的txt文件，修改为自己想要保存的位置
        
        for i in range(num_file):
            if random.random() < val:
                val_file.write(f'D:/yolov11/YOLOv11-pt-master/Dataset/training/images/{i:06}.png\n')   # 修改为kitti数据集图片的位置，即txt文件里存的是图片的位置
            else:
                train_file.write(f'D:/yolov11/YOLOv11-pt-master/Dataset/training/images/{i:06}.png\n')         
    finally:
        val_file.close()
        train_file.close()

if __name__ == '__main__':
    splitdataset()