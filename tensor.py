import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 假设你有一个支持CUDA的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 计算每个元素所占用的字节数
bytes_per_element = 4  # 对于float32，每个元素占用4字节

# 估算要创建的Tensor的shape，以便占用大约4GB显存
# 假设显存占用是纯净的（不考虑PyTorch的内部开销）
# 4GB = 4 * 1024^3 bytes
target_bytes = 3 * (1024 ** 3)

# 计算需要的元素数量
num_elements = target_bytes // bytes_per_element

# 由于显存分配通常不是连续的，并且PyTorch和其他程序可能也在使用显存，
# 我们可能需要分配稍微多一点的显存才能确保达到目标
# 增加一个安全系数，比如10%
safety_factor = 1.10
num_elements *= safety_factor

# 计算Tensor的shape。这里我们假设Tensor是一个二维的，即矩阵形式。
# 你可以根据需要调整这个shape，比如改为三维或更多维度。
# 假设矩阵的一维大小为sqrt(num_elements)，这样另一个维度也是这个大小。
# 注意：这可能导致内存碎片，因为不是所有shape都能被有效利用。
sqrt_num_elements = int(np.sqrt(num_elements))
tensor_shape = (sqrt_num_elements, sqrt_num_elements)

# 在GPU上创建一个大的Tensor
big_tensor = torch.rand(tensor_shape, device=device, dtype=torch.float32)

# 打印Tensor的信息
print(f"Created tensor of shape {big_tensor.shape} on device {device}")
print(f"Estimated size: {big_tensor.numel() * bytes_per_element / (1024 ** 3):.2f} GB")

# 确保Tensor保持在内存中，不被垃圾回收
big_tensor = big_tensor.clone()

# 可以添加一些操作来确保Tensor被实际使用，比如求和
# sum_result = big_tensor.sum()
# print(sum_result)