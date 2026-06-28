import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from function.HD95 import hd95_binary_2d
from function.UEO import compute_UEO, compute_sUEO
from function.uc_score import compute_assd, tune_unc_thresholds
from metrics_ece import compute_uncertainty_ece_summary, format_uncertainty_ece_lines
from utils import save_imgs


def _native_uncertainty(gamma, v, alpha, beta):
    aleatoric = beta / (alpha - 1 + 1e-8)
    epistemic = beta / (v * (alpha - 1 + 1e-8))
    return torch.sqrt(aleatoric + epistemic)


def _minmax_norm(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin + eps)


def _binary_mask(x, threshold=0.5):
    return (np.asarray(x) >= threshold).astype(np.uint8)


def _collect_prediction_batches(loader, model, config):
    uncertainty_maps, pred_masks, target_masks = [], [], []
    model.eval()
    with torch.no_grad():
        for img, msk in tqdm(loader):
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            gamma, v, alpha, beta = model(img)
            if isinstance(gamma, tuple):
                gamma = gamma[0]
            uncertainty = _native_uncertainty(gamma, v, alpha, beta)

            prob_np = gamma.squeeze(1).detach().cpu().numpy()
            unc_np = uncertainty.squeeze(1).detach().cpu().numpy()
            target_np = msk.squeeze(1).detach().cpu().numpy()

            for prob, unc, target in zip(prob_np, unc_np, target_np):
                pred = _binary_mask(prob, config.threshold)
                target_bin = _binary_mask(target, 0.5)
                uncertainty_maps.append(_minmax_norm(unc))
                pred_masks.append(pred)
                target_masks.append(target_bin)
    return uncertainty_maps, pred_masks, target_masks


def val_one_epoch(val_loader, model, criterion, scheduler, epoch, logger, config):
    del criterion, scheduler, epoch
    uncertainty_maps, pred_masks, target_masks = _collect_prediction_batches(val_loader, model, config)
    best = tune_unc_thresholds(
        uncertainty_maps,
        pred_masks,
        target_masks,
        voxelspacing=getattr(config, "voxelspacing", None),
        include_sueo=False,
    )
    logger.info(
        "Validation threshold tuning: ASSD th={:.4f}, UEO-Dice th={:.4f}, UEO-IoU th={:.4f}".format(
            best["ASSD"]["best_th"],
            best["UEO_Dice"]["best_th"],
            best["UEO_IoU"]["best_th"],
        )
    )
    return best


def test_one_epoch(test_loader, model, criterion, logger, config, test_data_name=None):
    del criterion
    model.eval()
    outputs_dir = os.path.join(config.work_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    preds_flat, gts_flat = [], []
    ueo_dice_list, ueo_iou_list, sueo_list = [], [], []
    assd_list, hd95_list = [], []
    uncertainty_maps, error_masks = [], []

    threshold_unc = getattr(config, "unc_threshold_assd", None)
    if threshold_unc is None:
        threshold_unc = 0.5

    with torch.no_grad():
        for i, (img, msk) in enumerate(tqdm(test_loader)):
            img = img.cuda(non_blocking=True).float()
            msk = msk.cuda(non_blocking=True).float()

            gamma, v, alpha, beta = model(img)
            if isinstance(gamma, tuple):
                gamma = gamma[0]
            uncertainty = _native_uncertainty(gamma, v, alpha, beta)

            prob = gamma.squeeze(1).detach().cpu().numpy()
            gt = msk.squeeze(1).detach().cpu().numpy()
            unc = uncertainty.squeeze(1).detach().cpu().numpy()

            for b in range(prob.shape[0]):
                prob_np = prob[b]
                gt_bin = _binary_mask(gt[b], 0.5)
                pred_bin = _binary_mask(prob_np, config.threshold)
                unc_norm = _minmax_norm(unc[b])
                error_mask = (pred_bin != gt_bin).astype(np.uint8)
                unc_mask = (unc_norm > threshold_unc).astype(np.uint8)

                preds_flat.append(pred_bin.reshape(-1))
                gts_flat.append(gt_bin.reshape(-1))
                uncertainty_maps.append(unc_norm)
                error_masks.append(error_mask)

                ueo_dice_list.append(compute_UEO(error_mask, unc_norm, thresh=threshold_unc, normalize_unc=False, metric="dice"))
                ueo_iou_list.append(compute_UEO(error_mask, unc_norm, thresh=threshold_unc, normalize_unc=False, metric="iou"))
                sueo_list.append(compute_sUEO(error_mask, unc_norm, normalize_unc=False))

                assd_value = compute_assd(error_mask, unc_mask, voxelspacing=getattr(config, "voxelspacing", None))
                if np.isfinite(assd_value):
                    assd_list.append(float(assd_value))

                spacing = getattr(config, "voxelspacing", (1.0, 1.0)) or (1.0, 1.0)
                hd95_value = hd95_binary_2d(pred_bin, gt_bin, spacing=spacing)
                if np.isfinite(hd95_value):
                    hd95_list.append(float(hd95_value))

                if i % config.save_interval == 0:
                    save_imgs(img, gt, prob, i, outputs_dir + "/", config.datasets, config.threshold, test_data_name=test_data_name)

    y_pred = np.concatenate(preds_flat) if preds_flat else np.array([], dtype=np.uint8)
    y_true = np.concatenate(gts_flat) if gts_flat else np.array([], dtype=np.uint8)
    if y_pred.size == 0:
        raise RuntimeError("No test predictions were collected.")

    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    accuracy = float(tn + tp) / max(float(np.sum(confusion)), 1.0)
    recall = float(tp) / max(float(tp + fn), 1.0)
    dice = float(2 * tp) / max(float(2 * tp + fp + fn), 1.0)
    iou = float(tp) / max(float(tp + fp + fn), 1.0)

    summary = {
        "Dice": dice,
        "IoU": iou,
        "Accuracy": accuracy,
        "Recall": recall,
        "HD95": float(np.mean(hd95_list)) if hd95_list else float("nan"),
        "UEO_Dice": float(np.mean(ueo_dice_list)) if ueo_dice_list else float("nan"),
        "UEO_IoU": float(np.mean(ueo_iou_list)) if ueo_iou_list else float("nan"),
        "sUEO": float(np.mean(sueo_list)) if sueo_list else float("nan"),
        "ASSD": float(np.mean(assd_list)) if assd_list else float("nan"),
    }
    ece_summary = compute_uncertainty_ece_summary(
        uncertainty_maps,
        error_masks,
        n_bins=getattr(config, "ece_bins", 15),
        as_percent=True,
    )

    lines = ["EviRisk-Seg test metrics"]
    lines.extend([f"{key}: {value:.4f}" for key, value in summary.items()])
    lines.append(format_uncertainty_ece_lines(ece_summary))
    message = "\n".join(lines)
    print(message)
    logger.info(message)

    metrics_path = os.path.join(config.work_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write(message + "\n")
    logger.info(f"Metrics saved to: {metrics_path}")
    return summary
