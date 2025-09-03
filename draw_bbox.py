import os
import cv2

img_dir = r'D:\pycharm_proj\yolov11\kaist\kaist_wash_picture_train\visible'
label_dir = r'D:\pycharm_proj\yolov11\kaist\kaist_wash_picture_train\labels'
output_dir = r'D:\pycharm_proj\yolov11\kaist\kaist_wash_picture_train\results'

print("=== 开始绘图 ===")
print("图像目录:", img_dir)
print("标签目录:", label_dir)
print("输出目录:", output_dir)

os.makedirs(output_dir, exist_ok=True)

count = 0

for filename in os.listdir(img_dir):
    if not filename.endswith('.png'):
        continue

    print(f"\n处理图像: {filename}")
    img_path = os.path.join(img_dir, filename)
    label_path = os.path.join(label_dir, filename.replace('.png', '.txt'))

    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ 读取失败: {img_path}")
        continue
    height, width = image.shape[:2]

    if not os.path.exists(label_path):
        print(f"⚠️ 无标签: {label_path}")
        continue

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            cls, x_center, y_center, box_w, box_h = map(float, line.split())
            x_center *= width
            y_center *= height
            box_w *= width
            box_h *= height
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, "person", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    save_path = os.path.join(output_dir, filename)
    success = cv2.imwrite(save_path, image)
    if success:
        print(f"✅ 保存成功: {save_path}")
        count += 1
    else:
        print(f"❌ 保存失败: {save_path}")

print(f"\n🎉 所有处理完成，共保存 {count} 张图像。")
