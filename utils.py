import os
import random
import torch
import warnings
import numpy as np
from losses import DiceLoss

criteria = DiceLoss() 

def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')
    
    
## evalute the performance
def get_mask(seg_volume):
    seg_volume = seg_volume.detach().cpu().numpy()
    seg_volume = np.squeeze(seg_volume)
    wt_pred = seg_volume[0]
    tc_pred = seg_volume[1]
    et_pred = seg_volume[2]
    mask = np.zeros_like(wt_pred)
    mask[wt_pred > 0.5] = 2
    mask[tc_pred > 0.5] = 1
    mask[et_pred > 0.5] = 4
    mask = mask.astype("uint8")
    return mask


def eval_metrics(gt, pred):
    loss_wt = criteria(np.where(gt>0, 1, 0), np.where(pred>0, 1, 0))
    loss_ct = criteria(np.where(gt==1, 1, 0)+np.where(gt==4, 1, 0), np.where(pred==1, 1, 0)+np.where(pred==4, 1, 0))
    loss_et = criteria(np.where(gt==4, 1, 0), np.where(pred==4, 1, 0))
    
    return loss_wt, loss_et, loss_ct


def measure_dice_score(batch_pred, batch_y):
    pred = get_mask(batch_pred)
    gt   = get_mask(batch_y)
    loss_wt, loss_et, loss_ct = eval_metrics(gt, pred)
    score = (loss_wt+loss_et+loss_ct)/3.0
    
    return score, loss_wt, loss_et, loss_ct