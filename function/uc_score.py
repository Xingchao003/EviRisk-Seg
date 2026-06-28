import numpy as np
import torch
import scipy.ndimage as ndi
from typing import Tuple, Optional
try:
    from medpy.metric.binary import assd as medpy_assd
except ImportError:
    medpy_assd = None


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        # 标量或可转为数组的类型
        return np.asarray(x)
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
def _dice_score(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    inter = (a & b).sum()
    s = a.sum() + b.sum()
    if s == 0:  # 两者都空 → 视为完美重合
        return 1.0
    return (2.0 * inter) / (s + eps)

def _iou_score(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    if union == 0:  # 两者都空 → 视为完美重合
        return 1.0
    return inter / (union + eps)

def _sueo_score_soft(unc: np.ndarray, error_mask: np.ndarray, eps: float = 1e-8) -> float:
    """
    sUEO = 2 * sum(y_i * u_i) / (sum(y_i^2) + sum(u_i^2))
    这里 y_i 是二值 error mask，u_i 是 [0,1] 的不确定度值（已归一化）
    """
    y = error_mask.astype(np.float32)
    u = unc.astype(np.float32)
    num = 2.0 * np.sum(y * u)
    den = np.sum(y * y) + np.sum(u * u) + eps
    return num / den

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


def tune_unc_thresholds(entropy_maps, preds, targets, voxelspacing=None,
                        thr_min=0.05, thr_max=0.95, thr_steps=50,
                        include_sueo=True):
    """
    扫描阈值，分别为 ASSD(最小化)、UEO-Dice/UEO-IoU(最大化) 找最优阈值；
    同时计算 sUEO(无阈值，返回平均分数)。
    返回一个 dict，包含各指标的最佳阈值和阈值-分数曲线。
    """
    thresholds = torch.linspace(thr_min, thr_max, steps=thr_steps)
    assd_scores, ueo_dice_scores, ueo_iou_scores = [], [], []

    # 逐阈值扫描
    for thr in thresholds:
        thr_val = float(thr.item())
        total_assd, n_assd = 0.0, 0
        total_dice, n_dice = 0.0, 0
        total_iou,  n_iou  = 0.0, 0

        for ent, pred, tgt in zip(entropy_maps, preds, targets):

            ent_np = _to_np(ent)  # 支持 torch 或 numpy
            pred_np = _to_np(pred).astype(np.uint8)
            tgt_np = _to_np(tgt).astype(np.uint8)
            error_mask = (pred_np != tgt_np).astype(np.uint8)
            unc_mask   = (ent_np > thr_val).astype(np.uint8)

            # ASSD（可能返回 nan）：跳过 nan
            val_assd = compute_assd(error_mask, unc_mask, voxelspacing=voxelspacing)
            if not np.isnan(val_assd):
                total_assd += float(val_assd); n_assd += 1

            # UEO-Dice / IoU
            total_dice += _dice_score(error_mask, unc_mask); n_dice += 1
            total_iou  += _iou_score(error_mask,  unc_mask); n_iou  += 1

        assd_scores.append((total_assd / n_assd) if n_assd > 0 else float('inf'))
        ueo_dice_scores.append((total_dice / n_dice) if n_dice > 0 else 0.0)
        ueo_iou_scores.append((total_iou  / n_iou ) if n_iou  > 0 else 0.0)

    # 选最优阈值
    assd_best_idx     = int(torch.argmin(torch.tensor(assd_scores)).item())
    ueo_dice_best_idx = int(torch.argmax(torch.tensor(ueo_dice_scores)).item())
    ueo_iou_best_idx  = int(torch.argmax(torch.tensor(ueo_iou_scores)).item())

    best_th_assd     = float(thresholds[assd_best_idx].item())
    best_th_ueoDice  = float(thresholds[ueo_dice_best_idx].item())
    best_th_ueoIoU   = float(thresholds[ueo_iou_best_idx].item())

    result = {
        'thresholds': thresholds.tolist(),
        'ASSD': {
            'best_th': best_th_assd,
            'curve': assd_scores,
            'best_score': assd_scores[assd_best_idx],
        },
        'UEO_Dice': {
            'best_th': best_th_ueoDice,
            'curve': ueo_dice_scores,
            'best_score': ueo_dice_scores[ueo_dice_best_idx],
        },
        'UEO_IoU': {
            'best_th': best_th_ueoIoU,
            'curve': ueo_iou_scores,
            'best_score': ueo_iou_scores[ueo_iou_best_idx],
        }
    }
    if include_sueo:
        sueo_vals = []
        for ent, pred, tgt in zip(entropy_maps, preds, targets):
            ent_np = _to_np(ent)
            pred_np = _to_np(pred).astype(np.uint8)
            tgt_np = _to_np(tgt).astype(np.uint8)
            error_mask = (pred_np != tgt_np).astype(np.uint8)
            sueo_vals.append(_sueo_score_soft(ent_np, error_mask))
        sueo_mean = float(np.mean(sueo_vals)) if sueo_vals else 0.0
        result['sUEO'] = {'best_th': None, 'score': sueo_mean}
    return result
