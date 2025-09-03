import os
import xml.etree.ElementTree as ET

def get_object_names(xml_dir):
    name_set = set()

    for file in os.listdir(xml_dir):
        if file.endswith(".xml"):
            file_path = os.path.join(xml_dir, file)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    name = obj.find("name")
                    if name is not None:
                        name_set.add(name.text.strip())
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    return name_set

xml_directory = r"D:\pycharm_proj\yolov11\LLVIP\Annotations"
names = get_object_names(xml_directory)

print(f"共找到 {len(names)} 种 object <name> 标签：")
for name in sorted(names):
    print(f"- {name}")
