import os

import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted
import cv2
import tqdm
from skimage.io import imsave

class Datasets(Dataset):
    def __init__(self, root_path='images/data_demo', fusion_type='Far-Near', dataset_type='Lytro', batch_size=1, filter=None):
        super(Datasets, self).__init__()
        self.A_dir = os.path.join(root_path, fusion_type, dataset_type, 'ir')
        self.B_dir = os.path.join(root_path, fusion_type, dataset_type, 'vi')
        self.A_list = natsorted(os.listdir(self.A_dir))
        self.B_list = natsorted(os.listdir(self.B_dir))
        self.batch_size = batch_size

        # # # todo
        # self.A_list = self.A_list[4:]
        # self.B_list = self.B_list[4:]

        if filter is not None:
            filename = os.path.join('./index', fusion_type, dataset_type, '{}_{}.txt'.format(dataset_type, filter))
            # 打开文件并读取内容
            with open(filename, 'r') as file:
                content = file.read().strip()  # 读取文件内容并去除两端的空白字符（如换行符）

            # 尝试将读取到的字符串解析为Python列表
            try:
                # 使用eval函数将字符串转换为Python对象（注意：eval有安全风险，仅当你信任文件内容时使用）
                image_list = eval(content)
                print(image_list)
                self.A_list = image_list
                self.B_list = image_list
            except Exception as e:
                print(f"解析文件内容时发生错误: {e}")
                image_list = []  # 如果解析失败，则初始化一个空列表

            # 现在image_list是一个包含文件名的Python列表
        pass

    def __getitem__(self, index):
        image_name = self.A_list[index]
        A_path = os.path.join(self.A_dir, image_name)
        B_path = os.path.join(self.B_dir, image_name)

        A_img = self.image_read_plus(A_path)
        B_img = self.image_read_plus(B_path)


        A_img = A_img.squeeze(0)
        B_img = B_img.squeeze(0)

        assert A_img.shape == B_img.shape
        return A_img, B_img, image_name


    def __len__(self):
        return len(self.A_list)

    def image_read_plus(self, path):
        img = self.image_read(path, mode='RGB')[np.newaxis, ...] / 255.0
        img = np.transpose(img, (0, 3, 1, 2))
        img = img * 2 - 1
        scale = 32
        h, w = img.shape[2:]
        h = h - h % scale
        w = w - w % scale
        img = ((torch.FloatTensor(img))[:, :, :h, :w])
        return img

    def image_read(self, path, mode='RGB'):
        img_BGR = cv2.imread(path).astype('float32')
        assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
        if mode == 'RGB':
            img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        elif mode == 'GRAY':
            img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
        elif mode == 'YCrCb':
            img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        return img

class Datasets_Mutil(Dataset):
    def __init__(self, root_path='images/data_demo', fusion_type='Far-Near', dataset_type='Lytro', batch_size=1, filter=None):
        super(Datasets_Mutil, self).__init__()
        projec_path = os.path.join(root_path, fusion_type, dataset_type)
        self.file_list = []
        self.file_name_list = []
        for dir_name in sorted(os.listdir(projec_path)):
            self.file_list.append(os.path.join(projec_path, dir_name))
            self.file_name_list.append(dir_name)


        self.batch_size = batch_size

        # # # todo
        # self.A_list = self.A_list[4:]
        # self.B_list = self.B_list[4:]

        if filter is not None:
            # filename = os.path.join('./index', fusion_type, dataset_type, '{}_{}.txt'.format(dataset_type, filter))
            filename = os.path.join('./index', fusion_type, dataset_type, '{}_{}.txt'.format(dataset_type, filter))
            # 打开文件并读取内容
            with open(filename, 'r') as file:
                content = file.read().strip()  # 读取文件内容并去除两端的空白字符（如换行符）

            # 尝试将读取到的字符串解析为Python列表
            try:
                # 使用eval函数将字符串转换为Python对象（注意：eval有安全风险，仅当你信任文件内容时使用）
                image_list = eval(content)
                print(image_list)
                self.file_list =[os.path.join(projec_path, item) for item in image_list]
            except Exception as e:
                print(f"解析文件内容时发生错误: {e}")
                image_list = []  # 如果解析失败，则初始化一个空列表

            # 现在image_list是一个包含文件名的Python列表
        pass

    def __getitem__(self, index):
        image_name = self.file_list[index]
        image_name_out = str(image_name).split('/')[-1]
        image_name_list = []
        for sub_image_name in sorted(os.listdir(image_name)):
            sub_img_path = os.path.join(image_name, sub_image_name)
            sub_img = self.image_read_plus(sub_img_path)
            sub_img = sub_img.squeeze(0)
            image_name_list.append(sub_img)
        return image_name_list, image_name_out



    def __len__(self):
        return len(self.file_list)

    def image_read_plus(self, path):
        img = self.image_read(path, mode='RGB')[np.newaxis, ...] / 255.0
        img = np.transpose(img, (0, 3, 1, 2))
        img = img * 2 - 1
        scale = 32
        h, w = img.shape[2:]
        h = h - h % scale
        w = w - w % scale
        img = ((torch.FloatTensor(img))[:, :, :h, :w])
        return img

    def image_read(self, path, mode='RGB'):
        img_BGR = cv2.imread(path).astype('float32')
        assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
        if mode == 'RGB':
            img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        elif mode == 'GRAY':
            img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
        elif mode == 'YCrCb':
            img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        return img

def test_Datasets():
    root_path = '../../../../datasets/fusion_resize_scale'
    fusion_type = 'Havard'
    dataset_type = 'CT-MRI'
    batch_size = 1
    image_folder = '../exp/image_samples/temp2'

    data = Datasets(root_path=root_path, fusion_type=fusion_type, dataset_type=dataset_type,
                    batch_size=batch_size)
    data = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False)
    pbar = tqdm.tqdm(enumerate(data), total=len(data))

    for index, [A_img, B_img, img_name] in pbar:
        ir = A_img
        vi = B_img

        # mask = (vi > -1).float()
        #
        # ir_new = mask * ir

        mask_ir = (ir == -1).float()
        mask_vi = (vi == -1).float()

        vi_to_ir = mask_ir * vi
        ir_to_vi = mask_vi * ir

        ir_new = ir * (1 - mask_ir) + vi_to_ir
        vi_new = vi * (1 - mask_vi) + ir_to_vi

        vi_new += ( 1 - mask_vi ) * vi

        sample_set = vi_new

        b = sample_set.size(0)
        for i in range(0, b):
            sample = sample_set[i, ...].detach().cpu().squeeze().numpy()
            sample = np.transpose(sample, (1, 2, 0))
            sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
            sample = ((sample) * 255)
            sample = sample.astype(np.uint8)
            os.makedirs(image_folder, exist_ok=True)
            imsave(os.path.join(image_folder, "{}.{}".format(img_name[i].split(".")[0], 'png')), sample)

if __name__ == '__main__':

    root_path = '../../../../datasets/fusion_resize_scale/Multi-Image'
    fusion_type = 'OE-UE'
    dataset_type = 'SICE'
    batch_size = 2
    filter = 'regular_p1'

    data = Datasets_Mutil(root_path=root_path, fusion_type=fusion_type, dataset_type=dataset_type,
                    batch_size=batch_size, filter=filter)
    data = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False)
    pbar = tqdm.tqdm(enumerate(data), total=len(data))

    for index, [img_list, img_name] in pbar:
        print(img_name[0] + ">>>>>" + img_name[1])
        pass


