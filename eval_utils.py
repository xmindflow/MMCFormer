import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes, binary_opening, binary_closing
from losses import DiceLoss

criteria = DiceLoss()


def morph_op(msk_pred, j):
    for item in range(msk_pred.shape[0]):
        msk_pred[item] = binary_closing(msk_pred[item], structure=np.ones((j+1,j+1))).astype(msk_pred.dtype)
        msk_pred[item] = binary_fill_holes(msk_pred[item], structure=np.ones((j+1,j+1))).astype(msk_pred.dtype)
    return msk_pred

## evalute the performance
def get_mask(seg_volume, thresh):
    seg_volume = seg_volume.detach().cpu().numpy()
    seg_volume = np.squeeze(seg_volume)
    wt_pred = seg_volume[0]
    tc_pred = seg_volume[1]
    et_pred = seg_volume[2]
    mask = np.zeros_like(wt_pred)
    mask[wt_pred > thresh[0]] = 2
    mask[tc_pred > thresh[1]] = 1
    mask[et_pred > thresh[2]] = 4
    mask = mask.astype("uint8")
    return mask

def eval_metrics(gt, pred, wt_j, ct_j, et_j):
    
    wt_pred = np.where(pred>0, 1, 0)
    if (np.sum(wt_pred) >20) and (wt_j != None): wt_pred = morph_op(wt_pred, wt_j)
    loss_wt = criteria(np.where(gt>0, 1, 0), wt_pred)
    
    ct_pred = np.where(pred==1, 1, 0)+ np.where(pred==4, 1, 0)
    if (np.sum(ct_pred) >20) and (ct_j != None): ct_pred = morph_op(ct_pred, ct_j)
    loss_ct = criteria(np.where(gt==1, 1, 0)+np.where(gt==4, 1, 0), ct_pred)
    
    et_pred = np.where(pred==4, 1, 0)
    if (np.sum(et_pred) >20) and (et_j != None): tc_pred = morph_op(et_pred, ct_j)
    loss_et = criteria(np.where(gt==4, 1, 0), et_pred)
    
    return loss_wt, loss_et, loss_ct

def measure_dice_score(batch_pred, batch_y, thresh, wt_j, ct_j, et_j):
    pred = get_mask(batch_pred, thresh = thresh)
    gt   = get_mask(batch_y, thresh=[0.5, 0.5, 0.5])
    
    loss_wt, loss_et, loss_ct = eval_metrics(gt, pred, wt_j, ct_j, et_j)
    score = (loss_wt+loss_et+loss_ct)/3.0
    
    return score, loss_wt, loss_et, loss_ct