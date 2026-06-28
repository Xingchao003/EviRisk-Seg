import os
from datetime import datetime

from torchvision import transforms

from utils import *


class setting_config:
    """Default public configuration for EviRisk-Seg with a UNet backbone.

    Paths and common runtime options can be overridden by environment variables
    so the code does not contain machine-specific absolute paths.
    """

    network = os.getenv("EVIRISK_NETWORK", "unet_evi")
    datasets = os.getenv("EVIRISK_DATASET", "ISIC2017")
    data_path = os.path.join(os.getenv("EVIRISK_DATA_PATH", "./data/ISIC2017"), "")
    as_gray = os.getenv("EVIRISK_AS_GRAY", "false").strip().lower() in {"1", "true", "yes", "y"}

    model_config = {
        "num_classes": 1,
        "input_channels": 3,
        "depths": [2, 2, 2, 2],
        "depths_decoder": [2, 2, 2, 1],
        "drop_path_rate": 0.2,
        "load_ckpt_path": "",
    }

    loss = "evi_nodice"
    lambda_reg = float(os.getenv("EVIRISK_LAMBDA_REG", "0.5"))
    criterion = total_loss_no_dice(lambda_reg=lambda_reg)

    num_classes = 1
    input_size_h = int(os.getenv("EVIRISK_INPUT_H", "256"))
    input_size_w = int(os.getenv("EVIRISK_INPUT_W", "256"))
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = int(os.getenv("EVIRISK_NUM_WORKERS", "4"))
    seed = int(os.getenv("EVIRISK_SEED", "42"))
    world_size = None
    rank = None
    amp = False
    gpu_id = os.getenv("EVIRISK_GPU_ID", "0")
    batch_size = int(os.getenv("EVIRISK_BATCH_SIZE", "8"))
    epochs = int(os.getenv("EVIRISK_EPOCHS", "300"))
    threshold = float(os.getenv("EVIRISK_SEG_THRESHOLD", "0.5"))
    ece_bins = int(os.getenv("EVIRISK_ECE_BINS", "15"))
    unc_threshold_assd = None
    voxelspacing = None

    _default_work_dir = os.path.join(
        "results",
        f"{network}_{datasets}{epochs}{loss}_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    work_dir = os.path.join(os.getenv("EVIRISK_WORK_DIR", _default_work_dir), "")

    print_interval = 10
    val_interval = 30
    save_interval = int(os.getenv("EVIRISK_SAVE_INTERVAL", "1"))

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w),
    ])

    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w),
    ])

    opt = "AdamW"
    assert opt in ["Adadelta", "Adagrad", "Adam", "AdamW", "Adamax", "ASGD", "RMSprop", "Rprop", "SGD"]
    if opt == "AdamW":
        lr = float(os.getenv("EVIRISK_LR", "0.001"))
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False

    sch = "CosineAnnealingLR"
    T_max = int(os.getenv("EVIRISK_T_MAX", str(epochs)))
    eta_min = float(os.getenv("EVIRISK_ETA_MIN", "1e-5"))
    last_epoch = -1
