import argparse
import os
import sys

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from datasets.dataset import NPY_datasets
from engine_testevi_assd import test_one_epoch, val_one_epoch
from model_load import model_loading
from utils import *
from config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained EviRisk-Seg checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pth checkpoint.")
    parser.add_argument("--data-path", default=None, help="Dataset root with train/val/test subfolders.")
    parser.add_argument("--work-dir", default=None, help="Directory for logs and outputs.")
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--unc-threshold", type=float, default=None, help="Fixed uncertainty threshold. If omitted, validation ASSD is used.")
    parser.add_argument("--save-interval", type=int, default=None)
    return parser.parse_args()


def apply_cli_overrides(config, args):
    if args.data_path is not None:
        config.data_path = os.path.join(args.data_path, "")
    if args.work_dir is not None:
        config.work_dir = os.path.join(args.work_dir, "")
    if args.gpu_id is not None:
        config.gpu_id = str(args.gpu_id)
    if args.unc_threshold is not None:
        config.unc_threshold_assd = float(args.unc_threshold)
    if args.save_interval is not None:
        config.save_interval = int(args.save_interval)
    config.checkpoint = args.checkpoint
    return config


def main(config):
    print("#----------Creating logger----------#")
    sys.path.append(config.work_dir + "/")
    log_dir = os.path.join(config.work_dir, "log")
    outputs = os.path.join(config.work_dir, "outputs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger("test", log_dir)
    log_config_info(config, logger)

    print("#----------GPU init----------#")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print("#----------Preparing model----------#")
    model = model_loading(config).cuda()
    cal_params_flops(model, 256, logger)
    checkpoint = torch.load(config.checkpoint, map_location=torch.device("cpu"))
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)

    print("#----------Preparing dataset----------#")
    val_dataset = NPY_datasets(config.data_path, config, train=False, test=False, as_gray=config.as_gray)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)
    test_dataset = NPY_datasets(config.data_path, config, train=False, test=True, as_gray=config.as_gray)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)

    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    _ = GradScaler()

    print("#----------Validation threshold tuning----------#")
    if getattr(config, "unc_threshold_assd", None) is None:
        best = val_one_epoch(val_loader, model, criterion, scheduler, 0, logger, config)
        config.unc_threshold_assd = best["ASSD"]["best_th"]
    logger.info(f"Using uncertainty threshold = {config.unc_threshold_assd:.4f}")

    print("#----------Testing----------#")
    test_one_epoch(test_loader, model, criterion, logger, config)


if __name__ == "__main__":
    args = parse_args()
    config = apply_cli_overrides(setting_config, args)
    main(config)
