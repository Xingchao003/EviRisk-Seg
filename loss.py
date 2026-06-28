import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import BceDiceLoss


criterion = BceDiceLoss(wb=1, wd=1)
# ========== 你原始的 NLL（不改格式） ==========
def nll_loss(gamma, v, alpha, beta, y, criterion, eps=1e-8):
    """
    gamma: sigmoid(gamma) in [0,1]
    y: target in {0,1} or [0,1]
    v, alpha, beta: evidential params (positive, alpha>1)
    criterion: 你原本的分割损失（残差项）
    """
    two_beta = 2.0 * beta
    loss = criterion(gamma, y)  # <-- 保留你原始定义（不要动）

    log_term = 0.5 * torch.log(math.pi / (v + eps)) \
               - alpha * torch.log(two_beta * (1.0 + v) + eps) \
               + (alpha + 0.5) * torch.log(v * loss + two_beta * (1.0 + v) + eps)

    gamma_term = torch.lgamma(alpha + eps) - torch.lgamma(alpha + 0.5 + eps)
    return log_term + gamma_term


# ========== 你原始的 REG（不改） ==========
def evidence_regularizer(gamma, v, alpha, y):
    phi = 2.0 * v + alpha
    return torch.abs(y - gamma) * phi


# ========== 新增：对比/分离型不确定度对齐损失 ==========
def separation_uncertainty_loss(gamma, y, total_unc, margin=0.2,
                                detach_err=True, normalize_unc=True, eps=1e-8):
    """
    让 错误区域 mean(U) >= 正确区域 mean(U) + margin
    gamma: [B,1,...] in [0,1]
    y:     [B,1,...]
    total_unc: [B,1,...] positive
    """
    # soft error map
    E = (y - gamma).abs()
    if detach_err:
        E = E.detach()
        # E = E
    U = total_unc
    if normalize_unc:
        # per-sample normalize to [0,1] to make margin stable
        B = U.shape[0]
        U_flat = U.view(B, -1)
        umin = U_flat.min(dim=1, keepdim=True)[0]
        umax = U_flat.max(dim=1, keepdim=True)[0]
        U = ((U_flat - umin) / (umax - umin + eps)).view_as(U)

    w_e = E
    w_c = 1.0 - E

    B = U.shape[0]
    U_flat  = U.view(B, -1)
    we_flat = w_e.view(B, -1)
    wc_flat = w_c.view(B, -1)

    mu_e = (we_flat * U_flat).sum(dim=1) / (we_flat.sum(dim=1) + eps)
    mu_c = (wc_flat * U_flat).sum(dim=1) / (wc_flat.sum(dim=1) + eps)

    # want: mu_e - mu_c >= margin
    return F.relu(margin - (mu_e - mu_c)).mean()


# ========== 总损失：保留原 evi_loss + 新增 sep 项 ==========
class total_loss_no_dice_with_sep(nn.Module):
    def __init__(self,
                 lambda_reg=1.0,
                 lambda_u=0.1,
                 margin=0.2,
                 detach_err=True,
                 normalize_unc=True,
                 eps=1e-8):
        super().__init__()
        self.seg_criterion = criterion
        self.lambda_reg = float(lambda_reg)

        # 新增项参数
        self.lambda_u = float(lambda_u)
        self.margin = float(margin)
        self.detach_err = bool(detach_err)
        self.normalize_unc = bool(normalize_unc)
        self.eps = eps

        # cache for logging
        self._last_nll_mean = float("nan")
        self._last_reg_mean = float("nan")
        self._last_sep_mean = float("nan")

    def forward(self, gamma, v, alpha, beta, y):
        # 1) 你原来的 nll / reg（不改）
        nll = nll_loss(gamma, v, alpha, beta, y, criterion=self.seg_criterion, eps=self.eps)
        reg = evidence_regularizer(gamma, v, alpha, y)

        # 2) 计算 total uncertainty（你原来的公式）
        denom = (alpha - 1.0 + self.eps)
        aleatoric = beta / denom
        epistemic = beta / (v * denom + self.eps)
        total_unc = torch.sqrt(aleatoric + epistemic + self.eps)

        # 3) 新增：对比/分离损失
        sep = separation_uncertainty_loss(
            gamma=gamma,
            y=y,
            total_unc=total_unc,
            margin=self.margin,
            detach_err=self.detach_err,
            normalize_unc=self.normalize_unc,
            eps=self.eps
        )

        # 4) 缓存日志
        self._last_nll_mean = float(nll.mean().detach().cpu())
        self._last_reg_mean = float(reg.mean().detach().cpu())
        self._last_sep_mean = float(sep.detach().cpu())

        # 5) 总损失：原 evi_loss + 新增项
        evi_loss = nll.mean() + self.lambda_reg * reg.mean()
        total = evi_loss + self.lambda_u * sep
        return total

# ========== 总损失：保留原 evi_loss + 新增 sep 项 ==========
class total_loss_no_dice_with_sep_vc(nn.Module):
    def __init__(self,
                 lambda_reg=1.0,
                 lambda_u=0.1,
                 margin=0.2,
                 detach_err=True,
                 normalize_unc=True,
                 eps=1e-8):
        super().__init__()
        self.seg_criterion = criterion
        self.lambda_reg = float(lambda_reg)

        # 新增项参数
        self.lambda_u = float(lambda_u)
        self.margin = float(margin)
        self.detach_err = bool(detach_err)
        self.normalize_unc = bool(normalize_unc)
        self.eps = eps

        # cache for logging
        self._last_nll_mean = float("nan")
        self._last_reg_mean = float("nan")
        self._last_sep_mean = float("nan")

    def forward(self, gamma, v, alpha, beta, y):
        # 1) 你原来的 nll / reg（不改）
        nll = nll_loss(gamma, v, alpha, beta, y, criterion=self.seg_criterion, eps=self.eps)
        reg = evidence_regularizer(gamma, v, alpha, y)

        # 2) 计算 total uncertainty（你原来的公式）
        denom = (alpha - 1.0 + self.eps)
        aleatoric = beta / denom
        epistemic = beta / (v * denom + self.eps)
        total_unc = torch.sqrt(aleatoric + epistemic + self.eps)

        # 3) 新增：对比/分离损失
        sep = separation_uncertainty_loss(
            gamma=gamma,
            y=y,
            total_unc=total_unc,
            margin=self.margin,
            detach_err=self.detach_err,
            normalize_unc=self.normalize_unc,
            eps=self.eps
        )

        # 4) 缓存日志
        self._last_nll_mean = float(nll.mean().detach().cpu())
        self._last_reg_mean = float(reg.mean().detach().cpu())
        self._last_sep_mean = float(sep.detach().cpu())

        # 5) 总损失：原 evi_loss + 新增项
        evi_loss = nll.mean() + self.lambda_reg * reg.mean()
        total = evi_loss + self.lambda_u * sep
        var_loss = total_unc.mean()
        return total + 0.001 * var_loss