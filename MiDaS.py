import cv2
import torch
import urllib.request
import os
import cv2
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)
filename = "/data/sdb/luming/code/Evaluation-for-Image-Fusion-Linfeng-Tang/fusion_results/Ours/OE-UE/SICE_OU/SCIE_1.png"
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas = torch.hub.load("/home/luming/.cache/torch/hub/intel-isl_MiDaS_master", model_type, source='local')
# midas = torch.load("/home/luming/.cache/torch/hub/intel-isl_MiDaS_master", model_type)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transforms = torch.hub.load("/home/luming/.cache/torch/hub/intel-isl_MiDaS_master", "transforms", source='local')

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()

# plt.imshow(output)
# plt.show()

# 归一化到0-1范围（如果output不在这个范围内）
# 如果output已经是0-1范围，则跳过这一步
output_norm = (output - output.min()) / (output.max() - output.min())

# 转换为0-255范围（如果需要转换为uint8）
output_norm_uint8 = (output_norm * 255).astype(np.uint8)

# 应用颜色映射
output_colored = cv2.applyColorMap(output_norm_uint8, cv2.COLORMAP_MAGMA)

file_path = './depth_map_colored.png'

# 使用imwrite保存图像
cv2.imwrite(file_path, output_colored)

# 注意：OpenCV读取图像是BGR格式，而matplotlib默认是RGB格式
# 因此，如果你直接在matplotlib中显示OpenCV处理的图像，可能需要转换颜色通道顺序
# output_colored_rgb = cv2.cvtColor(output_colored, cv2.COLOR_BGR2RGB)

# # 显示图像
# plt.imshow(output_colored_rgb)
# plt.axis('off')  # 关闭坐标轴
# plt.title('Colored Depth Map')
# plt.show()