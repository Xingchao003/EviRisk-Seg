import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------- 1) 每图 BCE / Dice / BCE+Dice（返回 [B]） ----------
class BCELossPerImage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred/target: [B, C, ...] 或 [B, 1, H, W,(D)]
        # 逐像素 BCE，然后对每张图的所有像素求均值 => [B]
        bce_pix = F.binary_cross_entropy(pred, target, reduction='none')
        B = pred.size(0)
        return bce_pix.view(B, -1).mean(dim=1)

class DiceLossPerImage(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 返回每张图的 Dice loss 向量 [B]
        B = pred.size(0)
        p = pred.view(B, -1)
        t = target.view(B, -1)
        inter = (p * t).sum(dim=1)
        dice_score = (2 * inter + self.smooth) / (p.sum(dim=1) + t.sum(dim=1) + self.smooth)
        return 1.0 - dice_score  # [B]

class BceDiceLossPerImage(nn.Module):
    def __init__(self, wb=1.0, wd=1.0, smooth=1.0):
        super().__init__()
        self.bce = BCELossPerImage()
        self.dice = DiceLossPerImage(smooth=smooth)
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        # 返回 [B]，每张图一个 Δ
        b = self.bce(pred, target)    # [B]
        d = self.dice(pred, target)   # [B]
        return self.wb * b + self.wd * d  # [B]
# ---------- 2) NLL：把 Δ 做成“每图” ----------
# 备注：这里仍沿用你原先的 Student-t 公式，唯一改变是 Δ 的来源与形状
#       Δ_img: [B] -> 通过 view/broadcast 变成 [B,1,1,1,(1)] 与 v/alpha/beta 对齐
def nll_loss_per_image(gamma, v, alpha, beta, y, criterion_per_img):
    """
    gamma, v, alpha, beta, y: 形状与原来一致（通常是 [B,1,H,W,(D)]）
    criterion_per_img: 上面的 BceDiceLossPerImage 实例
    返回：逐像素 NLL 张量（与 gamma 同形状），方便外面再做 mean()
    """
    # 每图 Δ（不跨样本耦合）
    delta_vec = criterion_per_img(gamma, y)          # [B]
    # 展开为可广播的 [B, 1, 1, 1,(1)]
    expand_shape = [gamma.size(0)] + [1] * (gamma.dim() - 1)
    delta = delta_vec.view(*expand_shape)

    two_beta = 2.0 * beta
    # 原公式：log_term + gamma_term
    log_term = 0.5 * torch.log(torch.tensor(math.pi, device=gamma.device) / v) \
               - alpha * torch.log(two_beta * (1.0 + v)) \
               + (alpha + 0.5) * torch.log(v * delta + two_beta * (1.0 + v))
    gamma_term = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    return log_term + gamma_term
# ---------- 3) 证据正则保持不变（逐像素），外面统一取 mean ----------
def evidence_regularizer(gamma, v, alpha, y):
    phi = 2.0 * v + alpha
    return torch.abs(y - gamma) * phi
# ---------- 4) 总损失：用“每图 Δ”的 nll ----------
class new_total_loss_no_dice(nn.Module):
    def __init__(self, lambda_reg=1.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        # 用返回 [B] 的版本，避免批次均值耦合
        self.bcedice_perimg = BceDiceLossPerImage(wb=1.0, wd=1.0)

    def forward(self, gamma, v, alpha, beta, y):
        # NLL（逐像素张量）
        nll = nll_loss_per_image(gamma, v, alpha, beta, y, self.bcedice_perimg)
        # 证据正则（逐像素张量）
        reg = evidence_regularizer(gamma, v, alpha, y)

        # 记录 batch 级的监控指标
        self._last_nll_mean = float(nll.mean().detach().cpu())
        self._last_reg_mean = float(reg.mean().detach().cpu())

        # 最终标量损失
        return nll.mean() + self.lambda_reg * reg.mean()

