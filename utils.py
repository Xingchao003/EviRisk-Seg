import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric
def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)



def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr = config.lr,
            rho = config.rho,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr = config.lr,
            lr_decay = config.lr_decay,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay,
            amsgrad = config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr = config.lr,
            betas = config.betas,
            eps = config.eps,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr = config.lr,
            lambd = config.lambd,
            alpha  = config.alpha,
            t0 = config.t0,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            alpha = config.alpha,
            eps = config.eps,
            centered = config.centered,
            weight_decay = config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr = config.lr,
            etas = config.etas,
            step_sizes = config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr = config.lr,
            momentum = config.momentum,
            weight_decay = config.weight_decay,
            dampening = config.dampening,
            nesterov = config.nesterov
        )
    else: # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                        'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = config.step_size,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones = config.milestones,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma = config.gamma,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = config.T_max,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode = config.mode, 
            factor = config.factor, 
            patience = config.patience, 
            threshold = config.threshold, 
            threshold_mode = config.threshold_mode, 
            cooldown = config.cooldown, 
            min_lr = config.min_lr, 
            eps = config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = config.T_0,
            T_mult = config.T_mult,
            eta_min = config.eta_min,
            last_epoch = config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma**len(
                [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler



# def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
#     img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
#     img = img / 255. if img.max() > 1.1 else img
#     if datasets == 'retinal':
#         msk = np.squeeze(msk, axis=0)
#         msk_pred = np.squeeze(msk_pred, axis=0)
#     else:
#         msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
#         msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
#
#     plt.figure(figsize=(7,15))
#
#     plt.subplot(3,1,1)
#     plt.imshow(img)
#     plt.axis('off')
#
#     plt.subplot(3,1,2)
#     plt.imshow(msk, cmap= 'gray')
#     plt.axis('off')
#
#     plt.subplot(3,1,3)
#     plt.imshow(msk_pred, cmap = 'gray')
#     plt.axis('off')
#
#     if test_data_name is not None:
#         save_path = save_path + test_data_name + '_'
#     plt.savefig(save_path + str(i) +'.png')
#     plt.close()


# def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
#     # 处理图像数据
#     img_processed = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
#     img_processed = img_processed / 255. if img_processed.max() > 1.1 else img_processed
#
#     if datasets == 'retinal':
#         msk_processed = np.squeeze(msk, axis=0)
#         msk_pred_processed = np.squeeze(msk_pred, axis=0)
#     else:
#         msk_processed = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
#         msk_pred_processed = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)
#
#     # 创建三个不同的文件夹
#     img_dir = os.path.join(save_path, "images")
#     msk_dir = os.path.join(save_path, "ground_truth")
#     pred_dir = os.path.join(save_path, "predictions")
#
#     os.makedirs(img_dir, exist_ok=True)
#     os.makedirs(msk_dir, exist_ok=True)
#     os.makedirs(pred_dir, exist_ok=True)
#
#     # 确定文件名前缀
#     file_prefix = ""
#     if test_data_name is not None:
#         file_prefix = test_data_name + '_'
#
#     # # 分别保存三个图像
#     # # 保存原始图像
#     # plt.figure(figsize=(7, 7))
#     # plt.imshow(img_processed)
#     # plt.axis('off')
#     # plt.savefig(os.path.join(img_dir, f'{file_prefix}{i:04d}.png'))
#     # plt.close()
#
#     # # 保存 ground truth
#     # plt.figure(figsize=(7, 7))
#     # plt.imshow(msk_processed, cmap='gray')
#     # plt.axis('off')
#     # plt.savefig(os.path.join(msk_dir, f'{file_prefix}{i:04d}.png'))
#     # plt.close()
#
#     # 保存预测结果
#     plt.figure(figsize=(7, 7))
#     plt.imshow(msk_pred_processed, cmap='gray')
#     plt.axis('off')
#     plt.savefig(os.path.join(pred_dir, f'{file_prefix}{i:04d}.png'))
#     plt.close()

import os
import numpy as np
from PIL import Image  # pip install pillow

def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    # ------ 1) 预处理到 numpy ------
    img_np = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()   # HWC
    if img_np.max() > 1.1:   # 假如是 0~255 的 uint8/float
        img_np = img_np / 255.0
    img_np = np.clip(img_np, 0, 1)

    if datasets == 'retinal':
        msk_np = np.squeeze(msk, axis=0)
        msk_pred_np = np.squeeze(msk_pred, axis=0)
    else:
        msk_np = (np.squeeze(msk, axis=0) > 0.5).astype(np.uint8)
        msk_pred_np = (np.squeeze(msk_pred, axis=0) > threshold).astype(np.uint8)

    # ------ 2) 创建文件夹 ------
    img_dir  = os.path.join(save_path, "images")
    msk_dir  = os.path.join(save_path, "ground_truth")
    pred_dir = os.path.join(save_path, "predictions")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # ------ 3) 文件名前缀 ------
    file_prefix = (test_data_name + '_') if (test_data_name is not None) else ''
    fname = f'{file_prefix}{i:04d}.png'

    # ------ 4) 无白边、原尺寸保存（PIL 直接写入像素）------
    # 原图：如果是单通道就转 L，多通道就转 RGB
    if img_np.shape[-1] == 1:
        img_u8 = (img_np[..., 0] * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(img_u8, mode='L').save(os.path.join(img_dir, fname))
    else:
        img_u8 = (img_np * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(img_u8, mode='RGB').save(os.path.join(img_dir, fname))

    # GT 与预测：二值 0/1 转 0/255 的单通道
    msk_u8 = (msk_np.astype(np.uint8) * 255)
    pred_u8 = (msk_pred_np.astype(np.uint8) * 255)
    Image.fromarray(msk_u8, mode='L').save(os.path.join(msk_dir, fname))
    Image.fromarray(pred_u8, mode='L').save(os.path.join(pred_dir, fname))

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
    

class nDiceLoss_brats2023(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss_brats2023, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            #weight = [1] * self.n_classes
            weight = [0.1, 1.0, 1.0, 1.0]
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            #print(f"class{i}:", dice)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        #print("celoss:", loss_ce)
        loss_dice = self.diceloss(pred, target, softmax=True)
        #print("diceloss:", loss_dice)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss



class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss
criterion = BceDiceLoss(wb=1, wd=1)
def nll_loss(gamma, v, alpha, beta, y):
    two_beta = 2 * beta
    loss = criterion(gamma, y)
    log_term = 0.5 * torch.log(math.pi / v) - alpha * torch.log(two_beta * (1 + v)) + (alpha + 0.5) * torch.log(
        v * loss + two_beta * (1 + v))
    gamma_term = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return log_term + gamma_term


# 定义证据正则化器
def evidence_regularizer(gamma, v, alpha, y):
    phi = 2 * v + alpha
    return torch.abs(y - gamma) * phi


# 定义总损失函数
def total_loss(gamma, v, alpha, beta, y, lambda_reg=1):
    bcediceloss = criterion(gamma, y)
    nll = nll_loss(gamma, v, alpha, beta, y)
    reg = evidence_regularizer(gamma, v, alpha, y)
    evi_loss = nll.mean() + lambda_reg * reg.mean()
    return evi_loss + bcediceloss

class total_loss_no_dice(nn.Module):
    def __init__(self, lambda_reg=1,clamp=(0.05, 0.5)):
        super(total_loss_no_dice, self).__init__()
        self.bcedice = BceDiceLoss(wb=1, wd=1)
        self.lambda_reg = lambda_reg

    def forward(self, gamma, v, alpha, beta, y, epoch=None):
        nll = nll_loss(gamma, v, alpha, beta, y)
        reg = evidence_regularizer(gamma, v, alpha, y)


        # ↓↓↓ 新增：记录本 batch 的分量均值（供 val_one_epoch 收集）
        self._last_nll_mean = float(nll.mean().detach().cpu())
        self._last_reg_mean = float(reg.mean().detach().cpu())
        self._last_reg1_mean = self._last_reg_mean
        self._last_reg2_mean = 0.0
        # ↑↑↑

        evi_loss = nll.mean() + self.lambda_reg * reg.mean()
        return evi_loss
class total_loss_vc(nn.Module):
    def __init__(self, lambda_reg=1):
        super(total_loss_vc, self).__init__()
        self.bcedice = BceDiceLoss(wb=1, wd=1)
        self.lambda_reg = lambda_reg
    def forward(self, gamma, v, alpha, beta, y):
        dice_loss = self.bcedice(gamma, y)
        nll = nll_loss(gamma, v, alpha, beta, y)
        reg = evidence_regularizer(gamma, v, alpha, y)
        evi_loss = nll.mean() + self.lambda_reg * reg.mean()
        # ------------------- 计算不确定性 -------------------
        # 数据不确定性 (Aleatoric: E[σ²] = β/(α-1))
        aleatoric = beta / (alpha - 1 + 1e-8)  # +1e-8 防止除零
        # 模型不确定性 (Epistemic: Var[μ] = β/(v*(α-1)))
        epistemic = beta / (v * (alpha - 1 + 1e-8))
        # 总不确定性 (数据+模型)
        total_uncertainty = (aleatoric + epistemic) ** 0.5
        var_loss = total_uncertainty.mean()
        return evi_loss + 0.001 * var_loss


def evidence_regularizer_reg2(gamma, v, alpha, beta, y, eps: float = 1e-8):
    # 论文 Eq.(15): (y - gamma)^2 * (alpha*v/beta)
    return (y - gamma).pow(2) * (alpha * v) / (beta + eps)


class total_loss_no_dice_calibrated(nn.Module):
    def __init__(self, lambda_reg1=1.0, lambda_reg2=1.0, eps: float = 1e-8):
        super(total_loss_no_dice_calibrated, self).__init__()
        self.bcedice = BceDiceLoss(wb=1, wd=1)
        self.lambda_reg1 = lambda_reg1
        self.lambda_reg2 = lambda_reg2
        self.eps = eps
    def forward(self, gamma, v, alpha, beta, y):
        nll = nll_loss(gamma, v, alpha, beta, y)
        reg1 = evidence_regularizer(gamma, v, alpha, y)
        reg2 = evidence_regularizer_reg2(gamma, v, alpha, beta, y, eps=self.eps)

        # 记录分量均值
        self._last_nll_mean  = float(nll.mean().detach().cpu())
        self._last_reg1_mean = float(reg1.mean().detach().cpu())
        self._last_reg2_mean = float(reg2.mean().detach().cpu())

        evi_loss = nll.mean() + self.lambda_reg1 * reg1.mean() + self.lambda_reg2 * reg2.mean()
        return evi_loss

class total_loss_no_dice_calibrated_var(nn.Module):
    def __init__(self, lambda_reg1=1.0, lambda_reg2=1.0, a=30, eps: float = 1e-8):
        super(total_loss_no_dice_calibrated_var, self).__init__()
        self.bcedice = BceDiceLoss(wb=1, wd=1)
        self.lambda_reg1 = lambda_reg1
        self.lambda_reg2 = lambda_reg2
        self.eps = eps
        self.stop_epoch = a

    def get_var_weight(self, epoch):
        # 假设 epoch 从 1 开始计数
        if epoch <= self.stop_epoch:
            return 0.001
        else:
            return 0.0

    def forward(self, gamma, v, alpha, beta, y, epoch):
        nll = nll_loss(gamma, v, alpha, beta, y)
        reg1 = evidence_regularizer(gamma, v, alpha, y)
        reg2 = evidence_regularizer_reg2(gamma, v, alpha, beta, y, eps=self.eps)

        # 记录分量均值
        self._last_nll_mean  = float(nll.mean().detach().cpu())
        self._last_reg1_mean = float(reg1.mean().detach().cpu())
        self._last_reg2_mean = float(reg2.mean().detach().cpu())

        # ------------------- 计算不确定性 -------------------
        # 数据不确定性 (Aleatoric: E[σ²] = β/(α-1))
        aleatoric = beta / (alpha - 1 + 1e-8)  # +1e-8 防止除零
        # 模型不确定性 (Epistemic: Var[μ] = β/(v*(α-1)))
        epistemic = beta / (v * (alpha - 1 + 1e-8))
        # 总不确定性 (数据+模型)
        total_uncertainty = (aleatoric + epistemic) ** 0.5
        var_loss = total_uncertainty.mean()
        current_weight = self.get_var_weight(epoch)
        evi_loss = nll.mean() + self.lambda_reg1 * reg1.mean() + self.lambda_reg2 * reg2.mean()

        if epoch <= self.stop_epoch: return nll.mean() + self.lambda_reg1 * reg1.mean() + self.lambda_reg2 * reg2.mean() + 0.001 * var_loss
        else: return evi_loss

class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3, target) * 0.3 + self.bcedice(gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss


import scipy.ndimage as ndi
from typing import Tuple, Optional

try:
    from medpy.metric.binary import assd as medpy_assd
except ImportError:
    medpy_assd = None


def extract_surface(mask: np.ndarray) -> np.ndarray:
    """
    Extract the surface (boundary) voxel coordinates from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask array of shape (H, W) or (D, H, W).

    Returns
    -------
    coords : np.ndarray
        Array of shape (N, ndim) with the indices of surface voxels.
    """
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=structure)
    surface = mask ^ eroded
    coords = np.stack(np.nonzero(surface), axis=-1)
    return coords


def compute_assd(
    error_mask: np.ndarray,
    unc_mask: np.ndarray,
    voxelspacing: Optional[Tuple[float, ...]] = None
) -> float:
    """
    Compute the Average Symmetric Surface Distance (ASSD) between two binary masks.

    This function first tries MedPy; if unavailable or if MedPy fails,
    it falls back to a pure-Python KD-tree implementation using SciPy.

    Parameters
    ----------
    error_mask : np.ndarray
        Binary mask of segmentation errors (pred != gt).
    unc_mask : np.ndarray
        Binary mask of high-uncertainty regions.
    voxelspacing : tuple of float, optional
        Physical spacing of voxels along each axis, e.g. (z, y, x) or (y, x).

    Returns
    -------
    assd_value : float
        The average symmetric surface distance (in same units as voxelspacing).
    """
    # Attempt MedPy implementation
    if medpy_assd is not None:
        try:
            return medpy_assd(error_mask, unc_mask, voxelspacing=voxelspacing)
        except Exception:
            # Fallback to manual KD-tree implementation
            pass

    # Manual fallback using KDTree
    from scipy.spatial import cKDTree

    # Extract surface voxel coordinates
    coords1 = extract_surface(error_mask)
    coords2 = extract_surface(unc_mask)

    if coords1.size == 0 or coords2.size == 0:
        return float('nan')

    # Apply voxel spacing if provided
    if voxelspacing is not None:
        spacing = np.array(voxelspacing)
        coords1 = coords1 * spacing
        coords2 = coords2 * spacing

    # Build KD-trees and query nearest distances
    tree1 = cKDTree(coords1)
    tree2 = cKDTree(coords2)

    # distances from coords1 to coords2 and vice versa
    dists1, _ = tree2.query(coords1, k=1)
    dists2, _ = tree1.query(coords2, k=1)

    # Average symmetric surface distance
    assd_value = 0.5 * (dists1.mean() + dists2.mean())
    return assd_value


class myToTensor:
    def __init__(self):
        pass
    def __call__(self, data):
        image, mask = data

        if mask.max() > 1:
            mask = (mask > 0).astype("float32")
        return torch.tensor(image).permute(2,0,1), torch.tensor(mask, dtype=torch.float32).permute(2,0,1)
       

class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w
    def __call__(self, data):
        image, mask = data
        image_resized = TF.resize(image, [self.size_h, self.size_w], interpolation=InterpolationMode.BILINEAR)
        # 标签：最近邻插值
        mask_resized = TF.resize(mask, [self.size_h, self.size_w], interpolation=InterpolationMode.NEAREST)
        return image_resized, mask_resized
       

class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.hflip(image), TF.hflip(mask)
        else: return image, mask
            

class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.vflip(image), TF.vflip(mask)
        else: return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0,360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p
    def __call__(self, data):
        image, mask = data
        if random.random() < self.p: return TF.rotate(image,self.angle), TF.rotate(mask,self.angle)
        else: return image, mask 


class myNormalize:
    def __init__(self, data_name, train=True,test=True):
        if data_name == 'floodwater':
            if train:
                self.mean =[128.370, 127.930,116.355]
                self.std =[54.077,49.088,53.161]
            else:
                self.mean = [115.146,113.264,101.272]
                self.std = [53.860,48.334,51.939]
        elif data_name == 'ISIC2017':
            if train:
                self.mean = 157.160
                self.std = 27.46521
            elif test:
                self.mean = 155.553517
                self.std = 27.189516
            else:
                self.mean = 160.65346
                self.std = 27.103581
        elif data_name == 'CirrMRI':
            if train:
                self.mean = 46.338
                self.std = 47.943246
            elif test:
                self.mean = 44.1839
                self.std = 47.113178
            else:
                self.mean = 45.997
                self.std = 48.20117
        elif data_name == 'cvc':
            if train:
                self.mean = 72.570267
                self.std = 52.18128621
            elif test:
                self.mean = 73.293665
                self.std = 51.18492675
            else:
                self.mean = 71.7133614
                self.std = 50.717306
        elif data_name == 'ISIC2017_all':
            if train:
                self.mean = 156.89135285
                self.std = 28.00784292
            elif test:
                self.mean = 150.478851
                self.std = 28.72985286
            else:
                self.mean = 149.79387814
                self.std = 25.3004838
        elif data_name == 'CVC_kva':
            if train:
                self.mean = 94.947608
                self.std = 59.514761
            elif test:
                self.mean = 94.658350
                self.std = 59.7759476
            else:
                self.mean = 93.444762
                self.std = 59.00331
        elif data_name == 'ISIC2018_new':
            if train:
                self.mean = 155.3980255
                self.std = 28.06343
            elif test:
                self.mean = 154.82817
                self.std = 27.927865982
            else:
                self.mean = 155.2004
                self.std = 26.662733
        elif data_name == 'Br35H':
            if train:
                self.mean = 94.826996
                self.std = 79.988782
            elif test:
                self.mean = 66.6274866
                self.std = 56.47108823
            else:
                self.mean = 98.254402
                self.std = 83.7778131
        elif data_name == 'cell':
            if train:
                self.mean = 177.755577
                self.std = 38.79271039
            elif test:
                self.mean = 176.0767175
                self.std = 39.85621
            else:
                self.mean = 174.7615757
                self.std = 40.136314
        elif data_name == 'nasopharyngeal':
            if train:
                self.mean = 32.41496
                self.std = 44.97789
            elif test:
                self.mean = 32.556248
                self.std = 44.691173
            else:
                self.mean = 32.6463
                self.std = 45.3157
        elif data_name == 'DUT_OMRON':
            if train:
                self.mean = 119.09234
                self.std = 58.64011
            elif test:
                self.mean = 120.02508
                self.std = 58.94688
            else:
                self.mean = 117.59168
                self.std = 59.434936
    def __call__(self, data):
        img, msk = data
        img = img.astype(np.float32)
        img_normalized = (img-self.mean)/self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) 
                            / (np.max(img_normalized)-np.min(img_normalized))) * 255.
        msk_bin = (msk > 0).astype(np.uint8) * 255
        return img_normalized, msk_bin


import torchvision.transforms as T
class myRandomGaussianBlur(object):
    def __init__(self, p=0.5, kernel_size=5, sigma=(0.1, 2.0)):
        """
        p: 施加模糊的概率
        kernel_size: 卷积核大小（奇数），越大越模糊
        sigma: 高斯核标准差范围，可以是 float 或 (min, max)
        """
        self.p = p
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, sample):
        """
        sample 根据你自己的写法可能是：
        1) (img, mask) 的 tuple
        2) {'image': img, 'mask': mask} 的 dict

        下面先按 (img, mask) 写，如果你用的是 dict，我后面给你改法。
        """
        img, mask = sample  # 如果你是 dict，这行要换成：img, mask = sample['image'], sample['mask']

        if random.random() < self.p:
            # img 是 Tensor，形状 [C, H, W]
            img = self.blur(img)

        return img, mask  # 如果你是 dict，这里要换成：{'image': img, 'mask': mask}


from thop import profile	 ## 导入thop模块
def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')






def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0



def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                    test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
        # cv2.imwrite(test_save_path + '/'+case + '.png', prediction*255)
    return metric_list


import torch
