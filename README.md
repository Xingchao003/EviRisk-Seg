# EviRisk-Seg

This repository provides the core implementation of **EviRisk-Seg** for binary medical image segmentation. The default implementation uses a UNet backbone and includes training, inference, and uncertainty-aware evaluation utilities.

## Features

- EviRisk-Seg model implementation with UNet as the default backbone.
- Optional evidential backbones: ramMamba and UltraLight-VMUNet.
- Evidential training loss.
- Training and testing entry scripts.
- Segmentation metrics: Dice, IoU, Accuracy, Recall, and HD95.
- Uncertainty-region metrics: UEO-Dice, UEO-IoU, sUEO, and ASSD.
- Uncertainty-based ECE computed from the native uncertainty map.

## Data preparation

Place datasets under the `data/` folder. For example, the default path used by
`config_setting.py` is:

```text
data/ISIC2017/
```

Each dataset should follow this structure:

```text
data/
└── ISIC2017/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

The image and mask filenames should be sorted in the same order. Masks should
be binary segmentation masks, where foreground pixels are non-zero and
background pixels are zero.

You can also use a custom dataset path:

```bash
export EVIRISK_DATA_PATH=/path/to/dataset_root
```

## Installation

```bash
conda create -n evirisk python=3.8
conda activate evirisk
pip install -r requirements.txt
```

Install a PyTorch build suitable for your CUDA version if needed.

## Training

Runtime options are provided through environment variables to avoid machine-specific paths in code.

```bash
export EVIRISK_DATA_PATH=/path/to/dataset_root
export EVIRISK_WORK_DIR=results/unet_evirisk_isic2017
export EVIRISK_GPU_ID=0

python train_evi.py
```

Common optional variables:

```bash
export EVIRISK_DATASET=ISIC2017
export EVIRISK_NETWORK=unet_evi
export EVIRISK_BATCH_SIZE=8
export EVIRISK_EPOCHS=300
export EVIRISK_LR=0.001
export EVIRISK_LAMBDA_REG=0.5
```

Available network options:

```text
unet_evi
ramMamba_evi
UltraLight_VM_UNet_evi
```

For example, to train with UltraLight-VMUNet:

```bash
export EVIRISK_NETWORK=UltraLight_VM_UNet_evi
python train_evi.py
```

## Testing

```bash
python test_evi.py \
  --checkpoint results/unet_evirisk_isic2017/checkpoints/best.pth \
  --data-path /path/to/dataset_root \
  --work-dir results/unet_evirisk_isic2017_test \
  --gpu-id 0
```

If `--unc-threshold` is not provided, the validation set is used to select the ASSD-optimal uncertainty threshold, which is then fixed for testing.

## Notes

The default network is `unet_evi`. The optional ramMamba and UltraLight-VMUNet backbones may require additional CUDA-compatible dependencies such as `mamba-ssm` and `triton`.
