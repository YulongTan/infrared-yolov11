import os

label_dir = r'D:\pycharm_proj\yolov11\kaist\kaist_wash_picture_test\labels'
target_class = '0'
result_files = []

for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(label_dir, filename)
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue  # 跳过空行
                    first_col = line.strip().split()[0]
                    if first_col == target_class:
                        result_files.append(filename)
                        break  # 找到一个就不用再读这个文件了
        except Exception as e:
            print(f'Error reading {filename}: {e}')

print("以下文件中第一列含有类2：")
for file in result_files:
    print(file)
