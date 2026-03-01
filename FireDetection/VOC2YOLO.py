import xml.etree.ElementTree as ET
import os


def voc_to_yolo(xml_file, output_path):
    print(f"Processing: {xml_file}")
    if not os.path.exists(xml_file):
        print(f"File not found: {xml_file}")
        return

    with open(xml_file, 'rb') as f:
        first_bytes = f.read(100)
        print("File starts with:", first_bytes[:100])

    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w, h = int(size.find('width').text), int(size.find('height').text)

    with open(output_path, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in ["smoke", "black_smoke"]:
                class_id = 0
            elif class_name in ["fire", "sf"]:
                class_id = 1
            else:
                continue
            #class_id = 0 if class_name == 'smoke' else 1  # smoke: 0, fire: 1
            bndbox = obj.find('bndbox')
            xmin, ymin = int(bndbox.find('xmin').text), int(bndbox.find('ymin').text)
            xmax, ymax = int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# 示例调用
for xml_file in os.listdir('Annotations'):
    voc_to_yolo(f'Annotations/{xml_file}', f'labels/{xml_file.replace(".xml", ".txt")}')
