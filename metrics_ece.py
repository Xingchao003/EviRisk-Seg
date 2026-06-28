import numpy as np


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _minmax_norm(x, eps=1e-8):
    x = _to_numpy(x).astype(np.float64)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float64)
    return (x - xmin) / (xmax - xmin + eps)


def _squeeze_pair(uncertainty, error_mask):
    unc = _to_numpy(uncertainty)
    err = _to_numpy(error_mask)
    unc = np.squeeze(unc)
    err = np.squeeze(err)
    if unc.shape != err.shape:
        raise ValueError(f"uncertainty shape {unc.shape} != error_mask shape {err.shape}")
    return _minmax_norm(unc).reshape(-1), (err > 0).astype(np.float64).reshape(-1)


def compute_uncertainty_ece(uncertainty, error_mask, n_bins=15):
    """Uncertainty-based ECE for binary segmentation.

    Pixels are binned by native uncertainty after min-max normalization. For
    each bin, the mean uncertainty is compared with the empirical segmentation
    error frequency. Lower is better.
    """

    unc, err = _squeeze_pair(uncertainty, error_mask)
    total = len(unc)
    if total == 0:
        return 0.0

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    for i in range(int(n_bins)):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        in_bin = (unc >= lower) & (unc <= upper) if i == 0 else (unc > lower) & (unc <= upper)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        avg_unc = float(np.mean(unc[in_bin]))
        err_rate = float(np.mean(err[in_bin]))
        ece += (count / total) * abs(err_rate - avg_unc)
    return float(ece)


def compute_uncertainty_ece_summary(uncertainty_maps, error_masks, n_bins=15, as_percent=True):
    per_image = [
        compute_uncertainty_ece(unc, err, n_bins=n_bins)
        for unc, err in zip(uncertainty_maps, error_masks)
    ]
    if not per_image:
        global_ece = 0.0
        mean_ece = 0.0
        std_ece = 0.0
    else:
        unc_all = np.concatenate([_squeeze_pair(u, e)[0] for u, e in zip(uncertainty_maps, error_masks)])
        err_all = np.concatenate([_squeeze_pair(u, e)[1] for u, e in zip(uncertainty_maps, error_masks)])
        global_ece = compute_uncertainty_ece(unc_all, err_all, n_bins=n_bins)
        mean_ece = float(np.mean(per_image))
        std_ece = float(np.std(per_image, ddof=1)) if len(per_image) > 1 else 0.0

    scale = 100.0 if as_percent else 1.0
    return {
        "n_images": len(per_image),
        "n_bins": int(n_bins),
        "uncertainty_ece_global": global_ece * scale,
        "uncertainty_ece_image_mean": mean_ece * scale,
        "uncertainty_ece_image_std": std_ece * scale,
        "unit": "%" if as_percent else "fraction",
    }


def format_uncertainty_ece_lines(summary):
    unit = summary.get("unit", "")
    return "\n".join([
        f"Uncertainty-ECE Global: {summary['uncertainty_ece_global']:.4f}{unit}",
        (
            "Uncertainty-ECE Image Mean: "
            f"{summary['uncertainty_ece_image_mean']:.4f} ± "
            f"{summary['uncertainty_ece_image_std']:.4f}{unit}"
        ),
        f"Images: {summary['n_images']}, Bins: {summary['n_bins']}",
    ])
