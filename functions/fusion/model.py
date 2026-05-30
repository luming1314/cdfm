import os

import torch
from torch import nn
import torchvision

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

import resizer
from functions.fusion.baseline.Tardal.Tardal import Generator
from resizer import Resizer
from functions.fusion.pytorch_colors import rgb_to_ycbcr
import numpy as np
from skimage.io import imsave
import cv2
from kornia.losses import SSIMLoss
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
ssim_loss = SSIMLoss(window_size=11, reduction='none')
from omegaconf import OmegaConf
class Fusion(nn.Module):
    def __init__(self, ref_img):
        super(Fusion, self).__init__()
        self.ref_img = ref_img


    def get_residual(self, image):
        return self.ref_img - image

class Fusion_PLUS(nn.Module):
    def __init__(self, ref_img, down_N=2):
        super(Fusion_PLUS, self).__init__()
        self.sobelconv = Sobelxy().eval()
        self.A_img = ref_img[0]
        self.B_img = ref_img[1]
        h ,w = self.A_img.shape[2:]
        shape = (1, 3, h, w)
        shape_d = (1, 3, int(h / down_N), int(w / down_N))
        self.down = Resizer(shape, 1 / down_N).cuda()
        self.up = Resizer(shape_d, down_N).cuda()
        # self.A_img_ilvr = self.up(self.down(self.A_img))
        # self.B_img_ilvr = self.up(self.down(self.B_img))
        # self.fus_img_ilvr = self.up(self.down(0.5 * self.A_img + 0.5 * self.B_img))
        #
        # A_img_Y = rgb_to_ycbcr_Y(self.A_img)
        # B_img_Y = rgb_to_ycbcr_Y(self.B_img)
        #
        # self.A_img_Y_grad = self.sobelconv(A_img_Y)
        # self.B_img_Y_grad = self.sobelconv(B_img_Y)
        #
        # self.ref_max_grad = torch.max(self.A_img_Y_grad, self.B_img_Y_grad)


    # def get_residual(self, image, k_i):
    #     # return 0.5 * (self.A_img - image) + 0.5 * (self.B_img - image)
    #     # image_ilvr = self.up(self.down(image))
    #     # return self.A_img_ilvr - image_ilvr
    #     # return self.A_img - image + self.B_img - image
    #      return k_i - image
    def get_residual_pixel_A(self, image):
        # image_ilvr = self.up(self.down(image))
        # return self.A_img_ilvr - image_ilvr
         return self.A_img - image
    def get_residual_pixel_B(self, image):
        # image_ilvr = self.up(self.down(image))
        # return self.B_img_ilvr - image_ilvr
         return self.B_img - image

    def get_residual_pixel_max(self, image):
        # image_ilvr = self.up(self.down(image))
        # return self.B_img_ilvr - image_ilvr
         return torch.max(self.A_img, self.B_img) - image

    def get_residual_pixel_ref(self, ir, vi, image):
        zeros = torch.zeros_like(ir)
        ones = torch.ones_like(ir)
        mean = torch.mean(ir)
        w = torch.where(ir > mean, ir, mean)
        for i in range(1):
            mean = torch.mean(w)
            w = torch.where(w > mean, w, mean)
        mask_lab = torch.where(w > torch.mean(w), ones, zeros)

        return (ir - image) * mask_lab + (1 - mask_lab) * (vi - image)

    def get_residual_pixel_ref2(self, ref, image):

        return self.up(self.down(ref)) - self.up(self.down(image))
        # return ref - image



    # def get_residual_em(self, ref_img, image):
    #     image_ilvr = self.up(self.down(image))
    #     ref_img_ilvr = self.up(self.down(ref_img))
    #     return ref_img_ilvr - image_ilvr
    #
    # def get_grad_residual(self, image):
    #     image = rgb_to_ycbcr_Y(image)
    #     image_grad = self.sobelconv(image)
    #     return self.ref_max_grad - image_grad


    def get_updown(self, image):

        return self.up(self.down(image))



class Fusion_VGG16(nn.Module):
    def __init__(self, ref_img, down_N=4):
        super(Fusion_VGG16, self).__init__()
        self.ref_img = ref_img
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.cuda()
        self.vgg16.eval()

        self.A_img = ref_img[0]
        self.B_img = ref_img[1]
        h ,w = self.A_img.shape[2:]
        shape = (1, 3, h, w)
        shape_d = (1, 3, int(h / down_N), int(w / down_N))
        self.down = Resizer(shape, 1 / down_N).cuda()
        self.up = Resizer(shape_d, down_N).cuda()

        self.features_A = self.vgg16.features(self.A_img)
        self.features_B = self.vgg16.features(self.B_img)

    def get_residual_pixel_A(self, image):
         return self.A_img - image
    def get_residual_pixel_B(self, image):
         return self.B_img - image




    def get_residual_feature_A(self, image):
        img_feature = self.vgg16.features(image)
        return self.features_A - img_feature


    def get_residual_feature_B(self, image):
        img_feature = self.vgg16.features(image)
        return self.features_B - img_feature

    def get_updown(self, image):

        return self.up(self.down(image))

class Fusion_ResNet_bak(nn.Module):
    def __init__(self, ref_img, down_N=2):
        super(Fusion_ResNet_bak, self).__init__()
        self.ref_img = ref_img
        self.resnet18 = models.resnet18(pretrained=True)


        self.resnet18.cuda()
        self.resnet18.eval()

        self.A_img = ref_img[0]
        self.B_img = ref_img[1]
        h, w = self.A_img.shape[2:]
        shape = (1, 3, h, w)
        shape_d = (1, 3, int(h / down_N), int(w / down_N))
        self.down = Resizer(shape, 1 / down_N).cuda()
        self.up = Resizer(shape_d, down_N).cuda()

        self.features_A = self.extract_feature(self.A_img)
        self.features_B = self.extract_feature(self.B_img)


        pass





    def extract_feature(self, x):
        x = self.resnet18.conv1(x)
        # x = self.resnet18.bn1(x)
        # x = self.resnet18.relu(x)
        # x = self.resnet18.maxpool(x)
        # x = self.resnet18.layer1(x)
        return x



    def get_residual_pixel_A(self, image):
        return self.A_img - image

    def get_residual_pixel_B(self, image):
        return self.B_img - image

    def get_residual_feature_A(self, image):
        img_feature = self.extract_feature(image)
        return self.features_A - img_feature

    def get_residual_feature_B(self, image):
        img_feature = self.extract_feature(image)
        return self.features_B - img_feature

    def get_residual_feature_A_lab(self, ref, image):
        ref_feature = self.extract_feature(ref)
        img_feature = self.extract_feature(image)
        return ref_feature - img_feature


    def get_updown(self, image):
        return self.up(self.down(image))

    def get_ssim_pixel(self, image):
        return ssim_loss(image, self.A_img) + ssim_loss(image, self.B_img)


class Fusion_Tardal(nn.Module):
    def __init__(self, ref_img, down_N=2):
        super(Fusion_Tardal, self).__init__()
        self.A_img = ref_img[0]
        self.B_img = ref_img[1]
        self.Tardal = Generator()
        self.Tardal.load_state_dict(torch.load(os.path.join('functions/fusion/baseline/Tardal/weight', 'tardal.pt'), map_location='cuda'))
        self.Tardal.cuda()
        self.Tardal.eval()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.cuda()
        self.resnet18.eval()

        self.ref_feat = self.extract_feature(self.A_img, self.B_img)

    def extract_feature(self, ir, vi):
        # ir = rgb_to_ycbcr_Y(ir)
        # vi = rgb_to_ycbcr_Y(vi)
        ir = ir.transpose(0,1)
        vi = vi.transpose(0,1)
        src = torch.cat([ir, vi], dim=1)
        f = self.Tardal.encoder(src)
        return f

    def get_residual(self, img):
        # img_y = rgb_to_ycbcr_Y(img)
        img = img.transpose(0, 1)
        src = torch.cat([img, img], dim=1)
        img_feat = self.Tardal.encoder(src)
        return self.ref_feat - img_feat

class Fusion_ResNet(nn.Module):
    def __init__(self, ref_img, down_N=2):
        super(Fusion_ResNet, self).__init__()
        # self.sobelconv = Sobelxy().eval()
        self.ref_img = ref_img
        # self.resnet18 = models.resnet18(pretrained=True)
        # self.resnet18.cuda()
        # self.resnet18.eval()
        self.A_img = ref_img[0]
        self.B_img = ref_img[1]
        #
        # h, w = self.A_img.shape[2:]
        # shape = (1, 3, h, w)
        # shape_d = (1, 3, int(h / down_N), int(w / down_N))
        # self.down = Resizer(shape, 1 / down_N).cuda()
        # self.up = Resizer(shape_d, down_N).cuda()
        # #
        # self.A_feat = self.extract_feature(self.A_img)
        # self.B_feat = self.extract_feature(self.B_img)
        # #
        # A_img_Y = rgb_to_ycbcr_Y(self.A_img)
        # B_img_Y = rgb_to_ycbcr_Y(self.B_img)
        #
        # self.A_img_Y_grad = self.sobelconv(A_img_Y)
        # self.B_img_Y_grad = self.sobelconv(B_img_Y)
        #
        # self.W_A = self.A_img_Y_grad.mean() / (self.A_img_Y_grad.mean() + self.B_img_Y_grad.mean())

        # self.autoencoder = self.get_autoencoder()

        # 创建三元组损失函数实例
        # self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1)
        pass

    def get_autoencoder(self):
        config = OmegaConf.load("configs/latent/vq-f4-config.yaml")
        model = self.load_model_from_config(config, "../../MMIF-DDFM-main/models/latent/vq-f4-model.ckpt")
        return model

    def load_model_from_config(self, config, ckpt):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt)  # , map_location="cpu")
        sd = pl_sd["state_dict"]
        model_auto = instantiate_from_config(config.model)
        m, u = model_auto.load_state_dict(sd, strict=False)
        model_auto.cuda()
        model_auto.eval()
        return model_auto



    def extract_feature(self, x):
        x = self.resnet18.conv1(x)
        return x

    def get_residual(self, img):
        img_feat = self.autoencoder(img)[0]
        A_img_feat = self.autoencoder(self.A_img)[0]
        B_img_feat = self.autoencoder(self.B_img)[0]
        return A_img_feat - img_feat + B_img_feat - img_feat



        # img_feat = self.extract_feature(img)
        # return self.A_feat - img_feat + self.B_feat - img_feat


    def get_residual_pixel(self, img):
        # ir = self.A_img
        # vi = self.B_img
        # return (ir - img + vi - img) / 2
        g = 0
        for img_lab in self.ref_img:
            g_t = img_lab - img
            g += g_t
        return  g / int(len(self.ref_img))


    def get_residual_pixel_grad(self, img):
        if not hasattr(self, 'sobelconv'):
            self.sobelconv = Sobelxy().eval().to(img.device)
            
        def get_grad(x):
            grads = [self.sobelconv(x[:, i:i+1]) for i in range(x.shape[1])]
            return torch.cat(grads, dim=1)

        g = 0
        img_grad = get_grad(img)
        for img_lab in self.ref_img:
            grad_lab = get_grad(img_lab)
            g_t = grad_lab - img_grad
            g += g_t
        return g / int(len(self.ref_img))




    def get_residual_pixel_max(self, img):
        ir = self.A_img
        vi = self.B_img

        # mask_ir = (ir == -1).float()
        # mask_vi = (vi == -1).float()
        #
        # vi_to_ir = mask_ir * vi
        # ir_to_vi = mask_vi * ir
        #
        # ir_new = ir * (1 - mask_ir) + vi_to_ir
        # vi_new = vi * (1 - mask_vi) + ir_to_vi

        # return ir_new - img

        mask = (vi == -1).float()

        return mask * (torch.max(ir ,vi ) - img)






        # return mask * ((ir - img + vi -img) / 2) + (1 - mask) * (torch.max(ir, vi) - img)
        # return  (1 - mask) * (torch.max(ir, vi) - img)



        # return  (1 - mask) * (torch.max(ir, vi) - img) + mask * (ir - img + vi - img)

    def get_residual_pixel_triplet(self, img):
        ir = self.A_feat
        vi = self.B_feat
        img = self.extract_feature(img)
        return self.triplet_loss(img, ir, (ir + vi) / 2) + self.triplet_loss(img, vi, (ir + vi) / 2)



    def get_residual_pixel_ref(self, img, ref):
        return ref - img

    def get_residual_pixel_grad_A(self, img):
        return self.A_img - img, self.W_A
    def get_residual_pixel_grad_B(self, img):
        return self.B_img - img, 1 - self.W_A

    def get_updown(self, image):
        return self.up(self.down(image))

    def get_residual_grad(self, img):
        return torch.max(self.A_img_Y_grad, self.B_img_Y_grad) - self.sobelconv(img[:, :1])







class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

        # self.weightx = nn.Parameter(data=kernelx, requires_grad=True).cuda()
        # self.weighty = nn.Parameter(data=kernely, requires_grad=True).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def image_read_plus(path):
    img = image_read(path, mode='RGB')[np.newaxis, ...] / 255.0
    img = np.transpose(img, (0, 3, 1, 2))
    img = img * 2 - 1
    scale = 32 * 4
    h, w = img.shape[2:]
    h = h - h % scale
    w = w - w % scale
    img = ((torch.FloatTensor(img))[:, :, :h, :w])
    return img

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def rgb_to_ycbcr_Y(img):
    x_0_hat_ycbcr = rgb_to_ycbcr(img) / 255  # (-1,1)

    x_0_hat_ycbcr = torch.unsqueeze((x_0_hat_ycbcr[:, 0, :, :]), 1)
    return x_0_hat_ycbcr

if __name__ == '__main__':
    sobelconv = Sobelxy()
    img1 = image_read_plus("../../images/fusion_demo/Far-Near/A/lytro-01.png")
    img2 = image_read_plus("../../images/fusion_demo/Far-Near/B/lytro-01.png")
    # sample = img1 + img2
    # h, w = img1.shape[2:]
    # shape = (1, 3, h, w)
    # down_N = 2
    # shape_d = (1, 3, int(h / down_N), int(w / down_N))
    # down = Resizer(shape, 1 / down_N)
    # up = Resizer(shape_d, down_N)
    # sample= up(down(0.5 * img1 + 0.5 * img2))

    img1 = img1[:,:1].cuda()
    img2 = img2[:,:1].cuda()

    sample_1 = sobelconv(img1)
    sample_2 = sobelconv(img2)


    # x_0_hat_ycbcr_1 = rgb_to_ycbcr(img1) / 255  # (-1,1)
    # x_0_hat_y_1 = torch.unsqueeze((x_0_hat_ycbcr_1[:, 0, :, :]), 1)
    # sample_1 = x_0_hat_y_1.cuda()
    # sample_1 = sobelconv(sample_1)

    # x_0_hat_ycbcr_2 = rgb_to_ycbcr(img2) / 255  # (-1,1)
    # x_0_hat_y_2 = torch.unsqueeze((x_0_hat_ycbcr_2[:, 0, :, :]), 1)
    # sample_2 = x_0_hat_y_2.cuda()
    # sample_2 = sobelconv(sample_2)

    sample = torch.max(sample_1, sample_2)
    # sample = sample_1


    sample = sample.detach().cpu().squeeze().numpy()
    # sample = np.transpose(sample, (1, 2, 0))
    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
    sample = ((sample) * 255)
    sample = sample.astype(np.uint8)
    os.makedirs('./img', exist_ok=True)
    imsave(os.path.join('./img', "{}.{}".format("lytro-01.png".split(".")[0], 'png')), sample)









