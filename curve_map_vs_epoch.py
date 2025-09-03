import re
import matplotlib.pyplot as plt

# 路径
log_file = r"D:\pycharm_proj\yolov11\LLVIP\train\train_infrared_log.txt"

# 初始化列表
epochs = []
box_loss, cls_loss, dfl_loss = [], [], []
precision, recall, map50, map = [], [], [], []

# 读取并解析日志
with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i]
    match_epoch = re.match(r"\s*(\d+)/\d+\s+[\d.]+G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
    if match_epoch:
        epoch = int(match_epoch.group(1))
        box = float(match_epoch.group(2))
        cls = float(match_epoch.group(3))
        dfl = float(match_epoch.group(4))
        epochs.append(epoch)
        box_loss.append(box)
        cls_loss.append(cls)
        dfl_loss.append(dfl)

        while i < len(lines) and 'precision' not in lines[i]:
            i += 1
        if i + 1 < len(lines):
            eval_line = lines[i + 1]
            match_eval = re.match(r"\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", eval_line)
            if match_eval:
                precision.append(float(match_eval.group(1)))
                recall.append(float(match_eval.group(2)))
                map50.append(float(match_eval.group(3)))
                map.append(float(match_eval.group(4)))
            else:
                precision.append(None)
                recall.append(None)
                map50.append(None)
                map.append(None)
        i += 1
    else:
        i += 1

# 找出 mAP 最大的位置
best_index = map.index(max(filter(lambda x: x is not None, map)))
print("📌 Best Epoch Metrics:")
print(f"Epoch     : {epochs[best_index]}")
print(f"Box Loss  : {box_loss[best_index]:.4f}")
print(f"Cls Loss  : {cls_loss[best_index]:.4f}")
print(f"DFL Loss  : {dfl_loss[best_index]:.4f}")
print(f"Precision : {precision[best_index]:.4f}")
print(f"Recall    : {recall[best_index]:.4f}")
print(f"mAP@50    : {map50[best_index]:.4f}")
print(f"mAP       : {map[best_index]:.4f}")

# ---------- 绘图部分 ----------

plt.figure(figsize=(16, 6))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, box_loss, label='Box Loss')
plt.plot(epochs, cls_loss, label='Cls Loss')
plt.plot(epochs, dfl_loss, label='DFL Loss')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 验证指标曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, precision, label='Precision')
plt.plot(epochs, recall, label='Recall')
plt.plot(epochs, map50, label='mAP@50')
plt.plot(epochs, map, label='mAP')
plt.title("Evaluation Metrics per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
