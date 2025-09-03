import glob
import random
import cv2
from tqdm import tqdm

dic = {'Car': 0,'Pedestrian': 1, 'Cyclist': 2}

def changeformat():
    
    img_path = 'D:\\yolov11\\YOLOv11-pt-master\\Dataset\\video\\tracking\\training\\image_02\\0000\\*.png'      # 修改为自己的 KITTI数据集图像位置
    label_path = 'D:\\yolov11\\YOLOv11-pt-master\\Dataset\\video\\tracking\\training\\label\\'         # 修改为自己的 KITTI数据集标签位置
    filename_list = glob.glob(img_path)
    save_path = 'D:\\yolov11\\YOLOv11-pt-master\\Dataset\\video\\tracking\\training\\label_process\\'                      # 修改为自己的 标签另存的位置

    for img_name in tqdm(filename_list, desc='Processing'):

        image_name = img_name[-10: -4]   # 000000 图片的名字

        label_file = label_path + image_name + '.txt'     # 根据图像名称查找对应标签
        savelabel_path = save_path + image_name + '.txt'  # 标签另存的文件
        
        with open(label_file, 'r') as f:
            labels = f.readlines()
        img = cv2.imread(img_name)
        h, w, c = img.shape
        dw = 1.0 / w
        dh = 1.0 / h        # 方便一会归一化

        for label in labels:
            label = label.split(' ')
            
            classname = label[1]
            if classname not in dic: continue  # 我忽略了kitti数据集中的misc和dontcare
            
            x1, y1, x2, y2 = label[4: 8]
            x1 = eval(x1)
            y1 = eval(y1)
            x2 = eval(x2)
            y2 = eval(y2)

            # 归一化处理
            bx = (x1 + x2) / 2.0 * dw
            by = (y1 + y2) / 2.0 * dh
            bw = (x2 - x1) * dw
            bh = (y2 - y1) * dh
            
            # 这里定义数据保存的精度
            bx = round(bx, 6)
            by = round(by, 6)
            bw = round(bw, 6) 
            bh = round(bh, 6)
            
            classindex = dic[classname]
            with open(savelabel_path, 'a+') as w:
                w.write(f'{classindex} {bx} {by} {bw} {bh}\n')
    
    print('Done convert!')

if __name__ == '__main__':
    changeformat()