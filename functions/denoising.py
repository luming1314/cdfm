import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os

# from .clip.base_clip import CLIPEncoder
# from .face_parsing.model import FaceParseTool
# from .anime2sketch.model import FaceSketchTool
# from .landmark.model import FaceLandMarkTool
# from .arcface.model import IDLoss

# from .arcface.model import IDLoss # todo
# from .arcface.model_i2i import IDLoss_i2i # todo
from .fusion.EM_onestep import EM_Initial, EM_onestep
from .fusion.model import Fusion, Fusion_VGG16, Fusion_PLUS, Fusion_ResNet, Fusion_Tardal # todo
import time

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def fusion_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, time_travel=False, ref_img=None, stop=100, domain="face"):
    """


    HP = EM_Initial(ref_img[0])
    lamb = 0.5
    rho_em = 0.001

    # fusion = Fusion(ref_img=ref_img)
    fusion = Fusion_ResNet(ref_img=ref_img)
    # fusion = Fusion_VGG16(ref_img=ref_img)

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    k_i = 0

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if time_travel:
            if domain == "face":
                repeat = 1
            elif domain == "imagenet":
                if 800 >= i >= 500:
                    repeat = 10
                else:
                    repeat = 1
        else:
            repeat = 1

        for idx in range(repeat):
            # if i >= 500:
            #     x0_t = ref_img[0] + ref_img[1]
            #     c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            #     c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            #     c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            #     c3 = (c3.log() * 0.5).exp()
            #     xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
            #     x0_t = x0_t.detach()
            #     xt_next = xt_next.detach()
            #
            #     x0_preds.append(x0_t)
            #     xs.append(xt_next)
            #
            #     if idx + 1 < repeat:
            #         bt = at / at_next
            #         xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
            #     continue
            xt.requires_grad = True

            with torch.no_grad():
                et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            if i >=-1:
                x_h_t = x0_t.clone().detach()
                x_0_ref, bfHP = EM_onestep(f_pre=x_h_t / 255,
                                                I=ref_img[0],
                                                V=ref_img[1],
                                                HyperP=HP, lamb=lamb, rho=rho_em)
                HP = bfHP
                # x0_t = x_0_ref
                # # x0_t = 1 * (x0_t - fusion.get_updown(x0_t)) + x_0_ref
                #
                # c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
                # c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
                # c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
                # c3 = (c3.log() * 0.5).exp()
                # xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)
                # x0_t = x0_t.detach()
                # xt_next = xt_next.detach()
                #
                # x0_preds.append(x0_t)
                # xs.append(xt_next)
                #
                # if idx + 1 < repeat:
                #     bt = at / at_next
                #     xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
                # continue


            residual = fusion.get_residual_pixel(x0_t, x_0_ref)
            k_i += 1
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]


            # kk = norm_grad.mean()
            # print('kk=={}'.format(kk))
            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            # xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)

            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()

            # l2 = l1 * 0.5  # Far_Near lytro
            l2 = l1 * 1  # Far_Near lytro
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()

            xt_ver = xt - rho * norm_grad
            xt_ver = xt_ver.detach()
            with torch.no_grad():
                et = model(xt_ver, t)
            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt_ver - et * (1 - at).sqrt()) / at.sqrt()
            xt_next = c1 * x0_t + c2 * xt_ver + c3 * torch.randn_like(x0_t)

            # if i != 0:
            #     xt_next -= rho * norm_grad

            # rho = at.sqrt() * 100
            # xt_next -= rho * norm_grad


            x0_t = x0_t.detach()
            xt_next = xt_next.detach()

            x0_preds.append(x0_t)
            xs.append(xt_next)

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]

        """



    # CDFM (Ours)
    # HP = EM_Initial(ref_img[0])
    # lamb = 0.5
    # rho_em = 0.001

    # fusion = Fusion(ref_img=ref_img)
    fusion = Fusion_ResNet(ref_img=ref_img)
    # fusion = Fusion_VGG16(ref_img=ref_img)

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    x0_preds_ver = []
    xs_ver = []

    k_i = 0

    # iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if time_travel:
            if domain == "face":
                repeat = 1
            elif domain == "imagenet":
                if 800 >= i >= 500:
                    repeat = 10
                else:
                    repeat = 1
        else:
            repeat = 1

        for idx in range(repeat):
            xt.requires_grad = True

            with torch.no_grad():
                et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            # if i >=-1:
            #     x_h_t = x0_t.clone().detach()
            #     x_0_ref, bfHP = EM_onestep(f_pre=x_h_t / 255,
            #                                     I=ref_img[0],
            #                                     V=ref_img[1],
            #                                     HyperP=HP, lamb=lamb, rho=rho_em)
            #     HP = bfHP

            residual = fusion.get_residual_pixel(x0_t) # 通用的

            # residual = fusion.get_residual_pixel(xt) # 直接度量x_t和c的距离，测试了相同参数下，效果不好

            # residual2 = fusion.get_residual_pixel_max(x0_t) # 医学图像 第一步，目前不成功
            # norm = torch.linalg.norm(residual) + torch.linalg.norm(residual2) 第二步

            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            # xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)

            # l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1)

            # l2 = l1 * 0.5  # Far_Near lytro
            l2 = l1 * 1  # Far_Near lytro 这个是论文的
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()

            xt_ver = xt - rho * norm_grad
            xt_ver = xt_ver.detach()

            # 记入第一步获取的 xt_ver 和 x0_t
            x0_preds_ver.append(x0_t.detach())
            xs_ver.append(xt_ver.detach())



            with torch.no_grad():
                et = model(xt_ver, t)
            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt_ver - et * (1 - at).sqrt()) / at.sqrt()
            xt_next = c1 * x0_t + c2 * xt_ver + c3 * torch.randn_like(x0_t)



            x0_t = x0_t.detach()
            xt_next = xt_next.detach()

            x0_preds.append(x0_t)
            xs.append(xt_next)

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)

    # return x0_preds, xs
    SAVE_NOISE_FLAG = True
    if SAVE_NOISE_FLAG:
        return [xs[-1]], [xs, x0_preds_ver, xs_ver, x0_preds] # 这是为了不改变别的地方妥协的

    return [xs[-1]], [x0_preds[-1]]
    


def fusion_ddim_diffusion_FreeDoM(x, seq, model, b, cls_fn=None, rho_scale=1.0, time_travel=False, ref_img=None, stop=100, domain="face"):
        """
        对比了Lytro数据集上，我们的方法与FreeDoM的区别
        1. l1 * 0.5 batch_size=10, 20轮， 结果FreeDoM效果差和DDFM类似
        2. l1 * 1 batch_size=10, 50轮 ，结果FreeDoM效果和我们的CDFM很接近了，但是仔细观看细节，我们还是要强于FreeDoM
        3. l1 * 1 batch_size=10, 100轮， 结果FreeDoM效果比50轮，细节更丰富了，但是类似就是 加权求和的结果，而且主观可能还不如50
        """

        # HP = EM_Initial(ref_img[0])
        # lamb = 0.5
        # rho_em = 0.001

        # fusion = Fusion(ref_img=ref_img)
        fusion = Fusion_ResNet(ref_img=ref_img)
        # fusion = Fusion_VGG16(ref_img=ref_img)

        # setup iteration variables
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        k_i = 0

        # iterate over the timesteps
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            if time_travel:
                if domain == "face":
                    repeat = 1
                elif domain == "imagenet":
                    if 800 >= i >= 500:
                        repeat = 10
                    else:
                        repeat = 1
            else:
                repeat = 1

            for idx in range(repeat):
                xt.requires_grad = True

                with torch.no_grad():
                    et = model(xt, t)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                # if i >=-1:
                #     x_h_t = x0_t.clone().detach()
                #     x_0_ref, bfHP = EM_onestep(f_pre=x_h_t / 255,
                #                                     I=ref_img[0],
                #                                     V=ref_img[1],
                #                                     HyperP=HP, lamb=lamb, rho=rho_em)
                #     HP = bfHP

                residual = fusion.get_residual_pixel(x0_t)
                norm = torch.linalg.norm(residual)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

                c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
                c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
                c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
                c3 = (c3.log() * 0.5).exp()
                xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)

                # l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
                l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1)

                l2 = l1 * 0.5  # Far_Near lytro
                # l2 = l1 * 1  # Far_Near lytro
                rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()

                xt_next -= rho * norm_grad
                x0_t = x0_t.detach()
                xt_next = xt_next.detach()

                x0_preds.append(x0_t)
                xs.append(xt_next)

                if idx + 1 < repeat:
                    bt = at / at_next
                    xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)

        # return x0_preds, xs
        return [xs[-1]], [x0_preds[-1]]






def FN_F_MFI_WHU_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, time_travel=False, ref_img=None, stop=100, domain="face"):
    # HP = EM_Initial(ref_img[0])
    # lamb = 0.5
    # rho_em = 0.001

    # fusion = Fusion(ref_img=ref_img)
    fusion = Fusion_ResNet(ref_img=ref_img)
    # fusion = Fusion_VGG16(ref_img=ref_img)

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    k_i = 0

    # iterate over the timesteps
    start_time = time.time()
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if time_travel:
            if domain == "face":
                repeat = 1
            elif domain == "imagenet":
                if 800 >= i >= 500:
                    repeat = 10
                else:
                    repeat = 1
        else:
            repeat = 1

        for idx in range(repeat):
            xt.requires_grad = True

            with torch.no_grad():
                et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            # if i >=-1:
            #     x_h_t = x0_t.clone().detach()
            #     x_0_ref, bfHP = EM_onestep(f_pre=x_h_t / 255,
            #                                     I=ref_img[0],
            #                                     V=ref_img[1],
            #                                     HyperP=HP, lamb=lamb, rho=rho_em)
            #     HP = bfHP
            residual = fusion.get_residual_pixel(x0_t)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            # xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)

            # l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1)

            # l2 = l1 * 0.5  # Far_Near lytro
            l2 = l1 * 1  # Far_Near lytro
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()

            xt_ver = xt - rho * norm_grad
            xt_ver = xt_ver.detach()

            wo_et_ver = False # 消融实验，去除重新估计et
            if not wo_et_ver:
                with torch.no_grad():
                    et = model(xt_ver, t)
                if et.size(1) == 6:
                    et = et[:, :3]

            x0_t = (xt_ver - et * (1 - at).sqrt()) / at.sqrt()
            xt_next = c1 * x0_t + c2 * xt_ver + c3 * torch.randn_like(x0_t)



            x0_t = x0_t.detach()
            xt_next = xt_next.detach()

            x0_preds.append(x0_t)
            xs.append(xt_next)

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
    end_time = time.time()

    # 计算并打印运行时间
    runtime = end_time - start_time
    print(f"代码块运行时间为：{runtime} 秒")

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]

def FN_F_MFI_WHU_ddim_diffusion_diff_energy_function (x, seq, model, b, cls_fn=None, rho_scale=1.0, time_travel=False, ref_img=None, stop=100, domain="face", energy_func_cls='1'):
    # HP = EM_Initial(ref_img[0])
    # lamb = 0.5
    # rho_em = 0.001

    # fusion = Fusion(ref_img=ref_img)
    fusion = Fusion_ResNet(ref_img=ref_img)
    # fusion = Fusion_VGG16(ref_img=ref_img)

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    k_i = 0

    # iterate over the timesteps
    start_time = time.time()
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if time_travel:
            if domain == "face":
                repeat = 1
            elif domain == "imagenet":
                if 800 >= i >= 500:
                    repeat = 10
                else:
                    repeat = 1
        else:
            repeat = 1

        for idx in range(repeat):
            xt.requires_grad = True

            with torch.no_grad():
                et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            # if i >=-1:
            #     x_h_t = x0_t.clone().detach()
            #     x_0_ref, bfHP = EM_onestep(f_pre=x_h_t / 255,
            #                                     I=ref_img[0],
            #                                     V=ref_img[1],
            #                                     HyperP=HP, lamb=lamb, rho=rho_em)
            #     HP = bfHP
            # residual = fusion.get_residual_pixel(x0_t)

            if energy_func_cls == '1':
                residual = fusion.get_residual_pixel(xt)
            elif energy_func_cls == '2':
                residual = fusion.get_residual_pixel_grad(x0_t)
            elif energy_func_cls == '3':
                residual = 0.5 * fusion.get_residual_pixel(x0_t) + 0.5 * fusion.get_residual_pixel_grad(x0_t)
            else:
                residual = fusion.get_residual_pixel(x0_t)

            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            # xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)

            # l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1)

            # l2 = l1 * 0.5  # Far_Near lytro
            l2 = l1 * 1  # Far_Near lytro
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()

            xt_ver = xt - rho * norm_grad
            xt_ver = xt_ver.detach()

            wo_et_ver = False # 消融实验，去除重新估计et
            if not wo_et_ver:
                with torch.no_grad():
                    et = model(xt_ver, t)
                if et.size(1) == 6:
                    et = et[:, :3]

            x0_t = (xt_ver - et * (1 - at).sqrt()) / at.sqrt()
            xt_next = c1 * x0_t + c2 * xt_ver + c3 * torch.randn_like(x0_t)



            x0_t = x0_t.detach()
            xt_next = xt_next.detach()

            x0_preds.append(x0_t)
            xs.append(xt_next)

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
    end_time = time.time()

    # 计算并打印运行时间
    runtime = end_time - start_time
    print(f"代码块运行时间为：{runtime} 秒")

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]

def FreeDoM_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, time_travel=False, ref_img=None, stop=100, domain="face"):
    print('FreeDoM start>>>>>>>')
    # HP = EM_Initial(ref_img[0])
    # lamb = 0.5
    # rho_em = 0.001

    # fusion = Fusion(ref_img=ref_img)
    fusion = Fusion_ResNet(ref_img=ref_img)
    # fusion = Fusion_VGG16(ref_img=ref_img)

    # setup iteration variables
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    k_i = 0

    # iterate over the timesteps
    start_time = time.time()
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')

        if time_travel:
            if domain == "face":
                repeat = 1
            elif domain == "imagenet":
                if 800 >= i >= 500:
                    repeat = 10
                else:
                    repeat = 1
        else:
            repeat = 1

        for idx in range(repeat):
            xt.requires_grad = True

            with torch.no_grad():
                et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            # if i >=-1:
            #     x_h_t = x0_t.clone().detach()
            #     x_0_ref, bfHP = EM_onestep(f_pre=x_h_t / 255,
            #                                     I=ref_img[0],
            #                                     V=ref_img[1],
            #                                     HyperP=HP, lamb=lamb, rho=rho_em)
            #     HP = bfHP
            residual = fusion.get_residual_pixel(x0_t)
            norm = torch.linalg.norm(residual)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

            c1 = at_next.sqrt() * (1 - at / at_next) / (1 - at)
            c2 = (at / at_next).sqrt() * (1 - at_next) / (1 - at)
            c3 = (1 - at_next) * (1 - at / at_next) / (1 - at)
            c3 = (c3.log() * 0.5).exp()
            xt_next = c1 * x0_t + c2 * xt + c3 * torch.randn_like(x0_t)

            # l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1).item()
            l1 = ((et * et).mean().sqrt() * (1 - at).sqrt() / at.sqrt() * c1)

            # l2 = l1 * 0.5  # Far_Near lytro
            l2 = l1 * 1  # Far_Near lytro
            rho = l2 / (norm_grad * norm_grad).mean().sqrt().item()

            xt_next -= rho * norm_grad

            # xt_ver = xt - rho * norm_grad
            # xt_ver = xt_ver.detach()
            #
            # wo_et_ver = False # 消融实验，去除重新估计et
            # if wo_et_ver:
            #     with torch.no_grad():
            #         et = model(xt_ver, t)
            #     if et.size(1) == 6:
            #         et = et[:, :3]

            # x0_t = (xt_ver - et * (1 - at).sqrt()) / at.sqrt()
            # xt_next = c1 * x0_t + c2 * xt_ver + c3 * torch.randn_like(x0_t)



            x0_t = x0_t.detach()
            xt_next = xt_next.detach()

            x0_preds.append(x0_t)
            xs.append(xt_next)

            if idx + 1 < repeat:
                bt = at / at_next
                xt = bt.sqrt() * xt_next + (1 - bt).sqrt() * torch.randn_like(xt_next)
    end_time = time.time()

    # 计算并打印运行时间
    runtime = end_time - start_time
    print(f"代码块运行时间为：{runtime} 秒")

    # return x0_preds, xs
    return [xs[-1]], [x0_preds[-1]]












# def far_near_fusion_ddim_diffusion(x, seq, model, b, cls_fn=None, rho_scale=1.0, stop=100, ref_img=None):
#     # fusion = Fusion(ref_img=ref_img)
#     fusion = Fusion_PLUS(ref_img=ref_img)
#
#     # setup iteration variables
#     n = x.size(0)
#     seq_next = [-1] + list(seq[:-1])
#     x0_preds = []
#     xs = [x]
#
#     # iterate over the timesteps
#     for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
#         t = (torch.ones(n) * i).to(x.device)
#         next_t = (torch.ones(n) * j).to(x.device)
#         at = compute_alpha(b, t.long())
#         at_next = compute_alpha(b, next_t.long())
#         xt = xs[-1].to('cuda')
#
#         xt.requires_grad = True
#
#         if cls_fn == None:
#             et = model(xt, t)
#         else:
#             print("use class_num")
#             class_num = 281
#             classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
#             et = model(xt, t, classes)
#             et = et[:, :3]
#             et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)
#
#         if et.size(1) == 6:
#             et = et[:, :3]
#
#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
#
#         residual = fusion.get_residual(x0_t)
#         norm = torch.linalg.norm(residual)
#         norm_grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]
#
#         eta = 0.5
#         c1 = (1 - at_next).sqrt() * eta
#         c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
#         xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x0_t) + c2 * et
#
#         # use guided gradient
#         rho = at.sqrt() * rho_scale
#         if not i <= stop:
#             xt_next -= rho * norm_grad
#
#         x0_t = x0_t.detach()
#         xt_next = xt_next.detach()
#
#         # x0_preds.append(x0_t.to('cpu'))
#         # xs.append(xt_next.to('cpu'))
#
#         x0_preds.append(x0_t)
#         xs.append(xt_next)
#
#     return [xs[-1]], [x0_preds[-1]]




