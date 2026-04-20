import os
import random
import shutil

# 设置路径
image_dir = "NEU-DET\IMAGES"      # 原始图片目录
label_dir = "NEU-DET\labels"      # 原始标签目录
train_img_dir = "dataset/images/train"
val_img_dir = "dataset/images/val"
train_label_dir = "dataset/labels/train"
val_label_dir = "dataset/labels/val"

# 创建目标文件夹
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 获取所有图片文件（假设格式为.jpg或.png）
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
# 打乱顺序
random.shuffle(image_files)

# 划分比例（例如80%训练，20%验证）
split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)

train_files = image_files[:split_index]
val_files = image_files[split_index:]

# 移动文件
for img in train_files:
    # 移动图片
    shutil.move(os.path.join(image_dir, img), os.path.join(train_img_dir, img))
    # 移动对应的标签（扩展名改为.txt）
    label_file = os.path.splitext(img)[0] + ".txt"
    if os.path.exists(os.path.join(label_dir, label_file)):
        shutil.move(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))
    else:
        print(f"警告：未找到标签文件 {label_file}")

for img in val_files:
    shutil.move(os.path.join(image_dir, img), os.path.join(val_img_dir, img))
    label_file = os.path.splitext(img)[0] + ".txt"
    if os.path.exists(os.path.join(label_dir, label_file)):
        shutil.move(os.path.join(label_dir, label_file), os.path.join(val_label_dir, label_file))
    else:
        print(f"警告：未找到标签文件 {label_file}")

print(f"划分完成！训练集: {len(train_files)} 张，验证集: {len(val_files)} 张。")