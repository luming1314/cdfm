import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
# from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
# from functions.denoising import clip_ddim_diffusion, parse_ddim_diffusion, sketch_ddim_diffusion, landmark_ddim_diffusion, arcface_ddim_diffusion
from functions.denoising import fusion_ddim_diffusion,FN_F_MFI_WHU_ddim_diffusion, FreeDoM_ddim_diffusion, FN_F_MFI_WHU_ddim_diffusion_diff_energy_function
import torchvision.utils as tvu

from guided_diffusion.unet import UNetModel
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth

# latent
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

from skimage.io import imsave
import einops

from resizer import Resizer


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, device=None):
        self.args = args
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = "fixedsmall"
        betas = get_beta_schedule(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, mode, data):
        cls_fn = None
        model_f = None
        model_i = None
        autoencoder = None


        if self.args.model_type == "imagenet":
            # get imagenet model
            imagenet_dict = {
                'type': 'openai', 
                'in_channels': 3, 
                'out_channels': 3, 
                'num_channels': 256, 
                'num_heads': 4, 
                'num_res_blocks': 2, 
                'attention_resolutions': '32,16,8', 
                'dropout': 0.0, 
                'resamp_with_conv': True, 
                'learn_sigma': True, 
                'use_scale_shift_norm': True, 
                'use_fp16': True, 
                'resblock_updown': True, 
                'num_heads_upsample': -1, 
                'var_type': 'fixedsmall', 
                'num_head_channels': 64, 
                'image_size': 256, 
                'class_cond': False, 
                'use_new_attention_order': False
                }
            model_i = create_model(**imagenet_dict)
            model_i.convert_to_fp16()
            ckpt = os.path.join('./weight', self.args.exp, "pre_model/256x256_diffusion_uncond.pt")
            model_i.load_state_dict(torch.load(ckpt, map_location=self.device))
            model_i.to(self.device)
            model_i.eval()
            model_i = torch.nn.DataParallel(model_i)
            model = model_i

        if self.args.latent_type == 'ldm':
            # Load latent Autoencoder
            autoencoder = self.get_autoencoder()

        if self.args.multi_image is not None:
            self.sample_sequence_mutil(model, autoencoder, cls_fn, mode, data)
        else:
            self.sample_sequence(model, autoencoder, cls_fn, mode, data)

    # Autoencoder
    def load_model_from_config(self, config, ckpt):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt)  # , map_location="cpu")
        sd = pl_sd["state_dict"]
        model_auto = instantiate_from_config(config.model)
        m, u = model_auto.load_state_dict(sd, strict=False)
        model_auto.to(self.device)
        model_auto.eval()
        return model_auto

    def get_autoencoder(self):
        config = OmegaConf.load("configs/latent/vq-f4-config.yaml")
        model = self.load_model_from_config(config, "../../MMIF-DDFM-main/models/latent/vq-f4-model.ckpt")
        return model


    def sample_sequence(self, model, autoencoder, cls_fn, mode, data):
        args = self.args
        # pbar = tqdm.tqdm(range(1, self.args.batch_size+1))
        pbar = tqdm.tqdm(enumerate(data), total=len(data))

        for index, [A_img, B_img, img_name] in pbar:
            A_img, B_img = A_img.to(self.device), B_img.to(self.device)
            b, c, h, w = B_img.shape
            self.args.batch_size = b
            ref_img = [A_img, B_img]

            # Flag = True
            # torch.manual_seed(self.args.seed)
            # if not Flag:
            #     x = torch.randn(1, 3, h, w, device=self.device)
            #     print("x=={}".format(x.mean()))
            # else:
            #     save_path = './fixed_seed/fixed_seed_tensor.pt'  # .pt是PyTorch模型的常见扩展名
            #     loaded_tensors_dict = torch.load(save_path, map_location=self.device)
            #     x = loaded_tensors_dict[f'tensor_h{h}_w{w}']
            #     print("x=={}".format(x.mean()))
            #
            # if self.args.batch_size > 1:
            #     torch.manual_seed(self.args.seed)
            #     y = x
            #     y_list = [y for _ in range(self.args.batch_size)]
            #     x = torch.cat(y_list, dim=0)
            #     print(">>>y=={}, x=={}".format(y.mean(), x.mean()))

            torch.manual_seed(self.args.seed)
            x = torch.randn(self.args.batch_size, 3, h, w, device=self.device)
            # x = torch.randn(1, 3, h, w, device=self.device)
            print("x=={}".format(x.mean()))
            Flag = True
            if Flag:
                save_path = './fixed_seed/fixed_seed_tensor.pt'  # .pt是PyTorch模型的常见扩展名
                loaded_tensors_dict = torch.load(save_path, map_location=self.device)
                x = loaded_tensors_dict[f'tensor_h{h}_w{w}']
                print("x_load=={}".format(x.mean()))
                pass
            if self.args.batch_size > 1:
                torch.manual_seed(self.args.seed)
                y = torch.randn(1, 3, h, w, device=self.device)
                print("y=={}, x=={}".format(y.mean(), x.mean()))

                if Flag:
                    save_path = './fixed_seed/fixed_seed_tensor.pt'  # .pt是PyTorch模型的常见扩展名
                    loaded_tensors_dict = torch.load(save_path, map_location=self.device)
                    y = loaded_tensors_dict[f'tensor_h{h}_w{w}']
                    print("y_load=={}".format(y.mean()))
                    pass

                y_list = [y for _ in range(self.args.batch_size)]
                x = torch.cat(y_list, dim=0)
                print(">>>y=={}, x=={}".format(y.mean(), x.mean()))

            if autoencoder is not None:
                # todo 目前还不可用
                A_img, B_img = autoencoder.encode(A_img)[0], autoencoder.encode(B_img)[0]
                h, w = B_img.shape[2:]
                ref_img = [A_img, B_img]
                x = torch.randn(self.args.batch_size, 3, h, w, device=self.device)
            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            if mode == "all_ddim":
                x, _ = self.sample_image_alogrithm_FN_F_MFI_WHU_ddim(x, model, last=False, cls_fn=cls_fn,
                                                                     rho_scale=args.rho_scale, ref_img=ref_img,
                                                                     stop=args.stop, domain=args.model_type)
            elif mode == "FreeDoM_ddim":
                x, _ = self.sample_image_alogrithm_FreeDoM_ddim(x, model, last=False, cls_fn=cls_fn,
                                                                     rho_scale=args.rho_scale, ref_img=ref_img,
                                                                     stop=args.stop, domain=args.model_type)

            elif mode == "fusion_ddim":
                # todo
                x, _ = self.sample_image_alogrithm_fusion_ddim(x, model, last=False, cls_fn=cls_fn,
                                                             rho_scale=args.rho_scale, ref_img=ref_img,
                                                             stop=args.stop, domain=args.model_type, img_name=img_name)

            if mode == "all_ddim_diff_energy_function":
                x, _ = self.sample_image_alogrithm_FN_F_MFI_WHU_ddim_Diff_Energy_Function(x, model, last=False, cls_fn=cls_fn,
                                                                     rho_scale=args.rho_scale, ref_img=ref_img,
                                                                     stop=args.stop, domain=args.model_type)
            elif mode == 'FN_F_ddim':
                if args.dataset_type == 'Lytro':
                    x, _ = self.sample_image_alogrithm_FN_F_Lytro_ddim(x, model, last=False, cls_fn=cls_fn,rho_scale=args.rho_scale, ref_img=ref_img, stop=args.stop, domain=args.model_type)
                elif  args.dataset_type == 'MFI-WHU':
                    x, _ = self.sample_image_alogrithm_FN_F_MFI_WHU_ddim(x, model, last=False, cls_fn=cls_fn, rho_scale=args.rho_scale, ref_img=ref_img,stop=args.stop, domain=args.model_type)
                else:
                    x, _ = self.sample_image_alogrithm_FN_F_MFI_WHU_ddim(x, model, last=False, cls_fn=cls_fn,
                                                                         rho_scale=args.rho_scale, ref_img=ref_img,
                                                                         stop=args.stop, domain=args.model_type)
                    print('Temporarily unsupported data!')

            # elif mode == 'far_near_fusion_ddim': # todo 这个暂时不用
            #     x, _ = self.sample_image_alogrithm_far_near_fusion_ddim(x, model, last=False, cls_fn=cls_fn, rho_scale=args.rho_scale, stop=args.stop, ref_img=ref_img)

            torch.cuda.empty_cache()
            sample_set = x[0]
            if autoencoder is not None:
                b = sample_set.size(0)
                for i in range(0, b):
                    kk = sample_set[i, ...].unsqueeze(0)
                    sample = autoencoder.decode(kk)
                    sample = sample.detach().cpu().squeeze().numpy()
                    sample = np.transpose(sample, (1, 2, 0))
                    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
                    sample = ((sample) * 255)
                    sample = sample.astype(np.uint8)
                    imsave(os.path.join(self.args.image_folder, "{}.{}".format(img_name[i].split(".")[0], 'png')), sample)
                # sample = autoencoder.decode(sample)
                # sample = (einops.rearrange(sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().detach().numpy().clip(0,255).astype(
                #     np.uint8)
            else:
                b = sample_set.size(0)
                for i in range(0, b):
                    sample = sample_set[i, ...].detach().cpu().squeeze().numpy()
                    sample = np.transpose(sample, (1, 2, 0))
                    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
                    sample = ((sample) * 255)
                    sample = sample.astype(np.uint8)
                    imsave(os.path.join(self.args.image_folder, "{}.{}".format(img_name[i].split(".")[0], 'png')), sample)

            SAVE_NOISE_FLAGE = False
            if SAVE_NOISE_FLAGE:
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
                seq_list = list(seq)
                seq_list.append(self.num_timesteps)
                seq_list.reverse()

                file_type = 'png'
                x_t_first_list, x0_t_first_list, x_t_second_list, x0_t_second_list = _
                length = len(x_t_first_list)
                for i in range(0, length):
                    b = x_t_first_list[i].size(0)
                    for j in range(0, b):
                        x_t_first = x_t_first_list[i][j, ...].detach().cpu().squeeze().numpy()
                        x_t_first = np.transpose(x_t_first, (1, 2, 0))
                        x_t_first = (x_t_first - np.min(x_t_first)) / (np.max(x_t_first) - np.min(x_t_first))
                        x_t_first = ((x_t_first) * 255)
                        x_t_first = x_t_first.astype(np.uint8)

                        x_t_first_path = os.path.join(self.args.image_folder, 'process', img_name[j].split(".")[0], 'x_t_first')
                        os.makedirs(x_t_first_path, exist_ok=True)
                        imsave(os.path.join(x_t_first_path, "x_{}_first.{}".format(seq_list[i], file_type)), x_t_first)

                        if i < length - 1:
                            x0_t_first = x0_t_first_list[i][j, ...].detach().cpu().squeeze().numpy()
                            x_t_second = x_t_second_list[i][j, ...].detach().cpu().squeeze().numpy()
                            x0_t_second = x0_t_second_list[i][j, ...].detach().cpu().squeeze().numpy()

                            x0_t_first = np.transpose(x0_t_first, (1, 2, 0))
                            x_t_second = np.transpose(x_t_second, (1, 2, 0))
                            x0_t_second = np.transpose(x0_t_second, (1, 2, 0))

                            x0_t_first = (x0_t_first - np.min(x0_t_first)) / (np.max(x0_t_first) - np.min(x0_t_first))
                            x_t_second = (x_t_second - np.min(x_t_second)) / (np.max(x_t_second) - np.min(x_t_second))
                            x0_t_second = (x0_t_second - np.min(x0_t_second)) / (np.max(x0_t_second) - np.min(x0_t_second))

                            x0_t_first = ((x0_t_first) * 255)
                            x_t_second = ((x_t_second) * 255)
                            x0_t_second = ((x0_t_second) * 255)

                            x0_t_first = x0_t_first.astype(np.uint8)
                            x_t_second = x_t_second.astype(np.uint8)
                            x0_t_second = x0_t_second.astype(np.uint8)

                            x0_t_first_path = os.path.join(self.args.image_folder, 'process', img_name[j].split(".")[0], 'x0_t_first')
                            x_t_second_path = os.path.join(self.args.image_folder, 'process', img_name[j].split(".")[0], 'x_t_second')
                            x0_t_second_path = os.path.join(self.args.image_folder, 'process', img_name[j].split(".")[0], 'x0_t_second')

                            os.makedirs(x0_t_first_path, exist_ok=True)
                            os.makedirs(x_t_second_path, exist_ok=True)
                            os.makedirs(x0_t_second_path, exist_ok=True)

                            imsave(os.path.join(x0_t_first_path, "x0_{}_first.{}".format(seq_list[i], file_type)), x0_t_first)
                            imsave(os.path.join(x_t_second_path, "x_{}_second.{}".format(seq_list[i], file_type)), x_t_second)
                            imsave(os.path.join(x0_t_second_path, "x0_{}_second.{}".format(seq_list[i], file_type)), x0_t_second)
    def sample_sequence_mutil(self, model, autoencoder, cls_fn, mode, data):
        args = self.args
        # pbar = tqdm.tqdm(range(1, self.args.batch_size+1))
        pbar = tqdm.tqdm(enumerate(data), total=len(data))

        for index, [img_list, img_name] in pbar:
            img_list = [img.to(self.device) for img in img_list]
            b,c,h,w = img_list[0].shape
            self.args.batch_size = b
            ref_img = img_list
            torch.manual_seed(self.args.seed)
            x = torch.randn(self.args.batch_size, 3, h, w, device=self.device)
            print("x=={}".format(x.mean()))
            Flag = True
            if Flag:
                save_path = './fixed_seed/fixed_seed_tensor.pt'  # .pt是PyTorch模型的常见扩展名
                loaded_tensors_dict = torch.load(save_path, map_location=self.device)
                x = loaded_tensors_dict[f'tensor_h{h}_w{w}']
                print("x_load=={}".format(x.mean()))
                pass
            if self.args.batch_size > 1:
                torch.manual_seed(self.args.seed)
                y = torch.randn(1, 3, h, w, device=self.device)
                print("y=={}, x=={}".format(y.mean(), x.mean()))

                if Flag:
                    save_path = './fixed_seed/fixed_seed_tensor.pt'  # .pt是PyTorch模型的常见扩展名
                    loaded_tensors_dict = torch.load(save_path, map_location=self.device)
                    y = loaded_tensors_dict[f'tensor_h{h}_w{w}']
                    print("y_load=={}".format(y.mean()))
                    pass

                y_list = [y for _ in range(self.args.batch_size)]
                x = torch.cat(y_list, dim=0)
                print(">>>y=={}, x=={}".format(y.mean(), x.mean()))
            if autoencoder is not None:
                # todo 目前还不可用
                A_img, B_img = autoencoder.encode(A_img)[0], autoencoder.encode(B_img)[0]
                h, w = B_img.shape[2:]
                ref_img = [A_img, B_img]
                x = torch.randn(self.args.batch_size, 3, h, w, device=self.device)
            # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
            if mode == "all_ddim":
                x, _ = self.sample_image_alogrithm_FN_F_MFI_WHU_ddim(x, model, last=False, cls_fn=cls_fn,
                                                                     rho_scale=args.rho_scale, ref_img=ref_img,
                                                                     stop=args.stop, domain=args.model_type)
            elif mode == "fusion_ddim":
                # todo
                x, _ = self.sample_image_alogrithm_fusion_ddim(x, model, last=False, cls_fn=cls_fn,
                                                             rho_scale=args.rho_scale, ref_img=ref_img,
                                                             stop=args.stop, domain=args.model_type, img_name=img_name)
            elif mode == 'FN_F_ddim':
                if args.dataset_type == 'Lytro':
                    x, _ = self.sample_image_alogrithm_FN_F_Lytro_ddim(x, model, last=False, cls_fn=cls_fn,rho_scale=args.rho_scale, ref_img=ref_img, stop=args.stop, domain=args.model_type)
                elif  args.dataset_type == 'MFI-WHU':
                    x, _ = self.sample_image_alogrithm_FN_F_MFI_WHU_ddim(x, model, last=False, cls_fn=cls_fn, rho_scale=args.rho_scale, ref_img=ref_img,stop=args.stop, domain=args.model_type)
                else:
                    x, _ = self.sample_image_alogrithm_FN_F_MFI_WHU_ddim(x, model, last=False, cls_fn=cls_fn,
                                                                         rho_scale=args.rho_scale, ref_img=ref_img,
                                                                         stop=args.stop, domain=args.model_type)
                    print('Temporarily unsupported data!')

            # elif mode == 'far_near_fusion_ddim': # todo 这个暂时不用
            #     x, _ = self.sample_image_alogrithm_far_near_fusion_ddim(x, model, last=False, cls_fn=cls_fn, rho_scale=args.rho_scale, stop=args.stop, ref_img=ref_img)

            torch.cuda.empty_cache()
            sample_set = x[0]
            if autoencoder is not None:
                b = sample_set.size(0)
                for i in range(0, b):
                    kk = sample_set[i, ...].unsqueeze(0)
                    sample = autoencoder.decode(kk)
                    sample = sample.detach().cpu().squeeze().numpy()
                    sample = np.transpose(sample, (1, 2, 0))
                    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
                    sample = ((sample) * 255)
                    sample = sample.astype(np.uint8)
                    imsave(os.path.join(self.args.image_folder, "{}.{}".format(img_name[i].split(".")[0], 'png')), sample)
                # sample = autoencoder.decode(sample)
                # sample = (einops.rearrange(sample, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().detach().numpy().clip(0,255).astype(
                #     np.uint8)
            else:
                b = sample_set.size(0)
                for i in range(0, b):
                    sample = sample_set[i, ...].detach().cpu().squeeze().numpy()
                    sample = np.transpose(sample, (1, 2, 0))
                    sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
                    sample = ((sample) * 255)
                    sample = sample.astype(np.uint8)
                    imsave(os.path.join(self.args.image_folder, "{}.{}".format(img_name[i].split(".")[0], 'png')), sample)

            SAVE_NOISE_FLAGE = True
            if SAVE_NOISE_FLAGE:
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
                seq_list = list(seq)
                seq_list.append(self.num_timesteps)
                seq_list.reverse()

                file_type = 'png'
                x_t_first_list, x0_t_first_list, x_t_second_list, x0_t_second_list = _
                length = len(x_t_first_list)
                for i in range(0, length):
                    b = x_t_first_list[i].size(0)
                    for j in range(0, b):
                        x_t_first = x_t_first_list[i][j, ...].detach().cpu().squeeze().numpy()
                        x_t_first = np.transpose(x_t_first, (1, 2, 0))
                        x_t_first = (x_t_first - np.min(x_t_first)) / (np.max(x_t_first) - np.min(x_t_first))
                        x_t_first = ((x_t_first) * 255)
                        x_t_first = x_t_first.astype(np.uint8)

                        x_t_first_path = os.path.join(self.args.image_folder, 'process', img_name[j].split(".")[0], 'x_t_first')
                        os.makedirs(x_t_first_path, exist_ok=True)
                        imsave(os.path.join(x_t_first_path, "x_{}_first.{}".format(seq_list[i], file_type)), x_t_first)

                        if i < length - 1:
                            x0_t_first = x0_t_first_list[i][j, ...].detach().cpu().squeeze().numpy()
                            x_t_second = x_t_second_list[i][j, ...].detach().cpu().squeeze().numpy()
                            x0_t_second = x0_t_second_list[i][j, ...].detach().cpu().squeeze().numpy()

                            x0_t_first = np.transpose(x0_t_first, (1, 2, 0))
                            x_t_second = np.transpose(x_t_second, (1, 2, 0))
                            x0_t_second = np.transpose(x0_t_second, (1, 2, 0))

                            x0_t_first = (x0_t_first - np.min(x0_t_first)) / (np.max(x0_t_first) - np.min(x0_t_first))
                            x_t_second = (x_t_second - np.min(x_t_second)) / (np.max(x_t_second) - np.min(x_t_second))
                            x0_t_second = (x0_t_second - np.min(x0_t_second)) / (np.max(x0_t_second) - np.min(x0_t_second))

                            x0_t_first = ((x0_t_first) * 255)
                            x_t_second = ((x_t_second) * 255)
                            x0_t_second = ((x0_t_second) * 255)

                            x0_t_first = x0_t_first.astype(np.uint8)
                            x_t_second = x_t_second.astype(np.uint8)
                            x0_t_second = x0_t_second.astype(np.uint8)

                            x0_t_first_path = os.path.join(self.args.image_folder, 'process', img_name[j].split(".")[0], 'x0_t_first')
                            x_t_second_path = os.path.join(self.args.image_folder, 'process', img_name[j].split(".")[0], 'x_t_second')
                            x0_t_second_path = os.path.join(self.args.image_folder, 'process', img_name[j].split(".")[0], 'x0_t_second')

                            os.makedirs(x0_t_first_path, exist_ok=True)
                            os.makedirs(x_t_second_path, exist_ok=True)
                            os.makedirs(x0_t_second_path, exist_ok=True)

                            imsave(os.path.join(x0_t_first_path, "x0_{}_first.{}".format(seq_list[i], file_type)), x0_t_first)
                            imsave(os.path.join(x_t_second_path, "x_{}_second.{}".format(seq_list[i], file_type)), x_t_second)
                            imsave(os.path.join(x0_t_second_path, "x0_{}_second.{}".format(seq_list[i], file_type)), x0_t_second)

    def sample_image_alogrithm_fusion_ddim(self, x, model, last=True, cls_fn=None, rho_scale=1.0, ref_img=None, stop=100,
                                         domain="face", img_name=None):
        time_travel = False
        if self.args.time_travel is not None:
            time_travel = True
        skip = self.num_timesteps // self.args.timesteps
        # if img_name is not None:
        #     path = os.path.join('configs', 'data_time', self.args.fusion_type, '{}-config.yaml'.format(self.args.dataset_type))
        #     config = OmegaConf.load(path)
        #     skip = self.num_timesteps // config.data[self.args.fusion_type][self.args.dataset_type][img_name[0].split('.')[0]]
        seq = range(0, self.num_timesteps, skip)

        x.requires_grad = True

        x = fusion_ddim_diffusion(x, seq, model, self.betas, cls_fn=cls_fn, rho_scale=rho_scale, ref_img=ref_img, time_travel=time_travel, stop=stop,
                                    domain=domain)

        if last:
            x = x[0][-1]
        return x


    def sample_image_alogrithm_FN_F_Lytro_ddim(self, x, model, last=True, cls_fn=None, rho_scale=1.0, ref_img=None, stop=100,
                                         domain="face"):
        time_travel = False
        if self.args.time_travel is not None:
            time_travel = True
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        x.requires_grad = True

        x = fusion_ddim_diffusion(x, seq, model, self.betas, cls_fn=cls_fn, rho_scale=rho_scale, ref_img=ref_img, time_travel=time_travel, stop=stop,
                                domain=domain)

        if last:
            x = x[0][-1]
        return x

    def sample_image_alogrithm_FN_F_MFI_WHU_ddim(self, x, model, last=True, cls_fn=None, rho_scale=1.0, ref_img=None, stop=100,
                                         domain="face"):
        time_travel = False
        if self.args.time_travel is not None:
            time_travel = True
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        x.requires_grad = True

        x = FN_F_MFI_WHU_ddim_diffusion(x, seq, model, self.betas, cls_fn=cls_fn, rho_scale=rho_scale, ref_img=ref_img, time_travel=time_travel, stop=stop,
                                    domain=domain)

        if last:
            x = x[0][-1]
        return x

    def sample_image_alogrithm_FN_F_MFI_WHU_ddim_Diff_Energy_Function(self, x, model, last=True, cls_fn=None, rho_scale=1.0, ref_img=None, stop=100,
                                         domain="face"):
        time_travel = False
        if self.args.time_travel is not None:
            time_travel = True
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        x.requires_grad = True

        x = FN_F_MFI_WHU_ddim_diffusion_diff_energy_function(x, seq, model, self.betas, cls_fn=cls_fn, rho_scale=rho_scale, ref_img=ref_img, time_travel=time_travel, stop=stop,
                                    domain=domain, energy_func_cls=self.args.energy_func_cls)

        if last:
            x = x[0][-1]
        return x

    def sample_image_alogrithm_FreeDoM_ddim(self, x, model, last=True, cls_fn=None, rho_scale=1.0, ref_img=None, stop=100,
                                         domain="face"):
        time_travel = False
        if self.args.time_travel is not None:
            time_travel = True
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)

        x.requires_grad = True

        x = FreeDoM_ddim_diffusion(x, seq, model, self.betas, cls_fn=cls_fn, rho_scale=rho_scale, ref_img=ref_img, time_travel=time_travel, stop=stop,
                                    domain=domain)

        if last:
            x = x[0][-1]
        return x






    # def sample_image_alogrithm_far_near_fusion_ddim(self, x, model, last=True, cls_fn=None, rho_scale=1.0, stop=100,
    #                                         ref_img=None):
    #     skip = self.num_timesteps // self.args.timesteps
    #     seq = range(0, self.num_timesteps, skip)
    #
    #     x.requires_grad = True
    #
    #     x = far_near_fusion_ddim_diffusion(x, seq, model, self.betas, cls_fn=cls_fn, rho_scale=rho_scale, stop=stop,
    #                                ref_img=ref_img)
    #
    #     if last:
    #         x = x[0][-1]
    #     return x





