import xml.etree.ElementTree as ET
import os

# 定义类别列表
CLASSES = [
    "crazing",          # 裂纹，ID为0
    "inclusion",        # 夹杂物，ID为1
    "patches",          # 斑块，ID为2
    "pitted_surface",   # 麻点，ID为3
    "rolled-in_scale",  # 氧化皮，ID为4
    "scratches"         # 划痕，ID为5
]

def convert_xml_to_yolo(xml_path, output_txt_path):
    """将单个XML文件转换为YOLO格式的TXT文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像的宽度和高度
        size = root.find('size')
        if size is None:
            print(f"警告: 在文件 {xml_path} 中未找到 <size> 标签，已跳过。")
            return
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        with open(output_txt_path, 'w') as out_file:
            # 遍历所有object标签
            for obj in root.findall('object'):
                # 获取类别名称并转换为索引
                name = obj.find('name').text
                if name not in CLASSES:
                    print(f"警告: 未知的类别名称 '{name}' 在文件 {xml_path} 中，已跳过。")
                    continue
                class_id = CLASSES.index(name)

                # 获取边界框的绝对坐标
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                # 计算YOLO所需的归一化坐标
                # 计算中心点坐标和宽高
                x_center = (xmin + xmax) / 2.0
                y_center = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin
                
                # 归一化到 [0, 1] 区间
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                width_norm = width / img_width
                height_norm = height / img_height

                # 写入文件
                out_file.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    except Exception as e:
        print(f"处理文件 {xml_path} 时出错: {e}")

def batch_convert(xml_dir, output_label_dir):
    """批量转换一个目录下的所有XML文件"""
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        txt_file = os.path.splitext(xml_file)[0] + '.txt'
        output_path = os.path.join(output_label_dir, txt_file)
        convert_xml_to_yolo(xml_path, output_path)

    print(f"转换完成！共处理 {len(xml_files)} 个文件，TXT文件已保存至: {output_label_dir}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请根据你的实际目录路径进行修改
    ANNOTATIONS_DIR = r"NEU-DET\ANNOTATIONS"   # 存放原始XML标注文件的目录
    LABELS_OUTPUT_DIR = r"NEU-DET\labels"      # 存放转换后TXT文件的目录

    batch_convert(ANNOTATIONS_DIR, LABELS_OUTPUT_DIR)