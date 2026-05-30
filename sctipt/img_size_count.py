import os
from PIL import Image
from collections import defaultdict


def get_image_size(image_path):
    """
    获取图片的尺寸
    """
    img = Image.open(image_path)
    return img.size


def count_image_sizes(folder_path):
    """
    统计文件夹中所有图片的尺寸、数量以及对应的文件名
    """
    size_counts = defaultdict(list)  # 使用defaultdict来自动初始化列表

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            image_path = os.path.join(folder_path, filename)
            size = get_image_size(image_path)
            size_counts[size].append(filename)  # 将文件名添加到对应尺寸的列表中

    return size_counts


def find_most_common_size(size_counts):
    """
    返回数量最多的尺寸以及对应的文件名列表（如果存在的话）
    """
    if not size_counts:  # 如果size_counts为空，则直接返回None和空列表
        return None, []

    max_count = max(size_counts.values(), key=len)  # 找到最大的列表长度
    if max_count == 0:  # 如果没有任何图片，也返回None和空列表
        return None, []

    most_common_sizes = [size for size, filenames in size_counts.items() if len(filenames) == max_count]  # 找到所有对应的尺寸
    if not most_common_sizes:  # 如果找不到对应的尺寸，返回None和空列表
        return None, []

    most_common_size = most_common_sizes[0]  # 取第一个尺寸（理论上这里应该只有一个）
    filenames = size_counts[most_common_size]  # 获取对应的文件名列表
    return most_common_size, filenames


# 设置图片文件夹路径
root_path = '../../../../datasets/fusion_resize_scale'
fusion_type = 'OE-UE' # IR-VI {TNO/RoadScene/M3FD/MSRS} Far-Near{Lytro/MFI-WHU} OE-UE{MEFB,SICE_OU}
dataset_type = 'SICE_OU'
folder_path = os.path.join(root_path, fusion_type, dataset_type, 'ir') # 请替换为你的图片文件夹路径


# 统计尺寸、数量和文件名
size_counts = count_image_sizes(folder_path)

# 找到数量最多的尺寸和对应的文件名列表
most_common_size, filenames = find_most_common_size(size_counts)

# 打印所有尺寸和数量（可选）
print("所有图片的尺寸和数量:")
for size, filenames in size_counts.items():
    print(f"尺寸: {size}, 数量: {len(filenames)}, 文件名: {filenames}")

# 打印数量最多的尺寸和对应的文件名列表
print(f"数量最多的尺寸是: {most_common_size}")
print(f"对应的文件名有:")
for filename in filenames:
    print(filename)