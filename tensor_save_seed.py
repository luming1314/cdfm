import torch
import numpy as np
import os
import logging
from PIL import Image
torch.set_printoptions(sci_mode=False)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置随机种子以获得可重复的随机数
seed = 1234
# set random seed
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True
# add device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info("Using device: {}".format(device))

# 图像数据集所在的根文件夹路径
dataset_root_folder = '../../../datasets/fusion_resize_scale'

# 创建一个空字典来存储tensors
tensors_dict = {}

# 使用os.walk递归遍历数据集中的所有图像
for dirpath, dirnames, filenames in os.walk(dataset_root_folder):
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 构建图像的完整路径
            image_path = os.path.join(dirpath, filename)

            # 打开图像并获取尺寸
            with Image.open(image_path) as img:
                width, height = img.size

                # 生成键名
                tensor_key = f'tensor_h{height}_w{width}'

                # 如果该尺寸的tensor已存在，则跳过以避免重复
                if tensor_key in tensors_dict:
                    continue

                    # 生成对应尺寸的tensor并存储到字典中
                torch.manual_seed(seed)
                tensor_value = torch.randn(1, 3, height, width, device=device)
                tensors_dict[tensor_key] = tensor_value

            # 打印tensors的信息或进行其他操作
for tensor_key, tensor_value in tensors_dict.items():
    print(f"{tensor_key} mean value: {tensor_value.mean()}")

# 如果需要保存这个字典，可以使用以下代码：
save_path = './fixed_seed/fixed_seed_tensor.pt'  # .pt是PyTorch模型的常见扩展名
torch.save(tensors_dict, save_path)
