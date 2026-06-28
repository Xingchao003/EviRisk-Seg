import numpy as np

def _minmax_norm(x, eps=1e-8):
    x = x.astype(np.float32)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    return (x - xmin) / (xmax - xmin + eps) if xmax > xmin else np.zeros_like(x, dtype=np.float32)

def _dice_binary(a, b, eps=1e-8):
    # a,b: {0,1} binary masks (np.uint8/np.bool_)
    inter = float((a & b).sum())
    s = float(a.sum() + b.sum())
    if s == 0:  # 两者都空 → 定义为完美一致
        return 1.0
    return (2.0 * inter) / (a.sum() + b.sum() + eps)

def _iou_binary(a, b, eps=1e-8):
    inter = float((a & b).sum())
    union = float((a | b).sum())
    if union == 0:  # 两者都空 → 定义为完美一致
        return 1.0
    return inter / (union + eps)

def compute_UEO(error_mask: np.ndarray,
                unc_map: np.ndarray,
                thresh: float = 0.5,
                normalize_unc: bool = True,
                metric: str = 'dice') -> float:
    """
    传统 UEO（带阈值）：先把不确定度图阈值成二值不确定掩码，再与错误掩码计算重叠（默认 Dice）。
    参数:
        error_mask: 0/1 错误区域掩码，形状(H,W) 或 (D,H,W)
        unc_map: 不确定度实值图(任意范围)，与 error_mask 同形状
        thresh: 不确定度阈值（在归一化到[0,1]空间后使用）
        normalize_unc: 是否对 unc_map 做 min-max 归一化到[0,1]
        metric: 'dice' 或 'iou'
    返回:
        标量重叠分数
    """
    assert error_mask.shape == unc_map.shape, "error_mask 与 unc_map 形状需一致"
    y = (error_mask > 0).astype(np.uint8)
    u = _minmax_norm(unc_map) if normalize_unc else unc_map.astype(np.float32)
    u_bin = (u > thresh).astype(np.uint8)

    if metric.lower() == 'dice':
        return _dice_binary(y, u_bin)
    elif metric.lower() == 'iou':
        return _iou_binary(y, u_bin)
    else:
        raise ValueError("metric 仅支持 'dice' 或 'iou'")

def compute_sUEO(error_mask: np.ndarray,
                 unc_map: np.ndarray,
                 normalize_unc: bool = True,
                 eps: float = 1e-8) -> float:
    """
    sUEO（无阈值软重叠）: 软 Dice = 2*Σ(y*u) / (Σ(y^2) + Σ(u^2))
    参数:
        error_mask: 0/1 错误区域掩码
        unc_map: 不确定度实值图(建议非负；本函数会做 min-max 到[0,1])
        normalize_unc: 是否对 unc_map 做 min-max 归一化到[0,1]
    返回:
        标量 sUEO
    """
    assert error_mask.shape == unc_map.shape, "error_mask 与 unc_map 形状需一致"
    y = (error_mask > 0).astype(np.float32)
    u = _minmax_norm(unc_map) if normalize_unc else unc_map.astype(np.float32)

    num = 2.0 * float((y * u).sum())
    den = float((y * y).sum() + (u * u).sum()) + eps
    return num / den
