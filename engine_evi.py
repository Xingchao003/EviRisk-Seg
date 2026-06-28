import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs, compute_assd
def predictive_entropy(p):
    return - (p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))

def compute_dice(pred, target):
    smooth = 1.0
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += 1
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        # images = fft_mask_layer(images)
        gamma, v, alpha, beta = model(images)

        loss = criterion(gamma, v, alpha, beta, targets,epoch)
        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())
        # —— 关键：从损失对象读取本 batch 的 NLL / 正则（已在 loss.forward 里缓存）——
        batch_nll = float(getattr(criterion, "_last_nll_mean", float('nan')))
        batch_reg = float(getattr(criterion, "_last_reg1_mean", float('nan')))
        batch_reg2 = float(getattr(criterion, "_last_reg2_mean", float('nan')))

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        # —— TensorBoard（可选）——

        writer.add_scalar('train/nll_batch', batch_nll, global_step=step)
        writer.add_scalar('uc_loss', loss, global_step=step)
        writer.add_scalar('train/reg_batch', batch_reg, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}, nll:{batch_nll:.4f}, reg:{batch_reg:.4f}, reg2:{batch_reg2:.4f}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step


def val_one_epoch(val_loader,
                    model,
                    criterion,
                    scheduler,
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds, gts, loss_list = [], [], []
    nll_list, reg_list, phi_list = [], [], []
    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            # img = fft_mask_layer(img)
            gamma, v, alpha, beta = model(img)

            loss = criterion(gamma, v, alpha, beta, msk,epoch)

            loss_list.append(loss.item())

            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(gamma) is tuple:
                gamma = gamma[0]
            gamma = gamma.squeeze(1).cpu().detach().numpy()
            preds.append(gamma)
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    y_pre = np.where(preds >= config.threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)


    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    if epoch == config.epochs:
        entropy_maps, bin_preds, bin_targets = [], [], []
        with torch.no_grad():
            for img, msk in tqdm(val_loader, desc='Collect for ASSD tun.'):
                img, msk = img.cuda().float(), msk.cuda().float()
                gamma, v, alpha, beta = model(img)
                # compute total uncertainty
                aleatoric = beta / (alpha - 1 + 1e-8)
                epistemic = beta / (v * (alpha - 1 + 1e-8))
                total_unc = torch.sqrt(aleatoric + epistemic)
                ent = total_unc.squeeze(1).cpu()
                # normalize uncertainty to [0,1]
                ent = (ent - ent.min()) / (ent.max() - ent.min() + 1e-8)
                gamma = gamma.squeeze(1).cpu()
                msk = msk.squeeze(1).cpu()
                for e, p, t in zip(ent, gamma, msk):
                    entropy_maps.append(e)
                    bin_preds.append(p)
                    bin_targets.append(t)
        best_th = tune_assd_threshold(entropy_maps, bin_preds, bin_targets,voxelspacing=getattr(config, 'voxelspacing', None))
        logger.info(f'Auto-tuned uncertainty threshold: {best_th:.3f}')
        print(f'Auto-tuned uncertainty threshold: {best_th:.3f}')
        # update config or save threshold
        config.unc_threshold = best_th

    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    epoch = 300
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            # img = fft_mask_layer(img)
            gamma, v, alpha, beta = model(img)

            loss = criterion(gamma, v, alpha, beta, msk,epoch)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(gamma) is tuple:
                gamma = gamma[0]
            gamma = gamma.squeeze(1).cpu().detach().numpy()
            preds.append(gamma)
            if i % config.save_interval == 0:
                save_imgs(img, msk, gamma, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
def tune_assd_threshold(entropy_maps, preds, targets, voxelspacing=None):
    """
    Compute best uncertainty threshold by minimizing ASSD between uncertain mask and error mask.
    entropy_maps: list[torch.Tensor] each shape (H,W)
    preds:        list[torch.Tensor] each shape (H,W) binary predictions
    targets:      list[torch.Tensor] each shape (H,W) binary targets
    voxelspacing: tuple for physical spacing passed to compute_assd
    """
    thresholds = torch.linspace(0.05, 0.95, steps=50)
    assd_scores = []
    for thr in thresholds:
        total_assd = 0.0
        count = 0
        for ent, pred, tgt in zip(entropy_maps, preds, targets):

            ent_np = ent.cpu().numpy()
            pred_mask = pred.cpu().numpy().astype(np.uint8)
            tgt_mask  = tgt.cpu().numpy().astype(np.uint8)
            error_mask = (pred_mask != tgt_mask).astype(np.uint8)
            unc_mask = (ent_np > thr.item()).astype(np.uint8)
            val = compute_assd(error_mask, unc_mask, voxelspacing=voxelspacing)
            if not np.isnan(val):
                total_assd += val
                count += 1
        avg_assd = total_assd / count if count>0 else float('inf')
        assd_scores.append(avg_assd)
    best_idx = torch.argmin(torch.tensor(assd_scores))
    return thresholds[best_idx].item()



