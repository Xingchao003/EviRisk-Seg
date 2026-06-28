import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask)
    return mask ^ eroded  # 边界

def _directed_surface_distances(src_surf: np.ndarray, dst_mask: np.ndarray, spacing=(1.0, 1.0)):
    # 近似“到边界”的距离：目标内外距离变换合并
    dt_out = distance_transform_edt(~dst_mask, sampling=spacing)
    dt_in  = distance_transform_edt(dst_mask,  sampling=spacing)
    boundary_dt = np.where(dst_mask, dt_in, dt_out)
    return boundary_dt[src_surf]

def hd95_binary_2d(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0)) -> float:
    """2D 二值掩码的 HD95；两者都空返回 0；仅一方有前景返回 inf。"""
    pred = pred.astype(bool); gt = gt.astype(bool)
    Sp, Sg = _surface(pred), _surface(gt)
    if not Sp.any() and not Sg.any():
        return 0.0
    if Sp.any() ^ Sg.any():
        return float('inf')
    d1 = _directed_surface_distances(Sp, gt,  spacing)
    d2 = _directed_surface_distances(Sg, pred, spacing)
    if d1.size == 0 or d2.size == 0:
        return float('inf')
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))
