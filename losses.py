import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

    
def get_current_consistency_weight(epoch, consistency = 10, consistency_rampup = 20.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * sigmoid_rampup(epoch, consistency_rampup)
  
    
def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def dice_loss(input, target):
    """soft dice loss"""
    eps = 1e-7
    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()

    return 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)


def gram_matrix(input):
    a, b, c, d, e = input.size()
    features = input.view(a * b, c * d * e)
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d * e)


def unet_Co_loss(batch_pred_full, batch_pred_missing, batch_y, epoch):
    loss_dict = {}
    
    loss_dict['ed_dc_loss']  = dice_loss(batch_pred_full[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['net_dc_loss'] = dice_loss(batch_pred_full[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_dc_loss']  = dice_loss(batch_pred_full[:, 2], batch_y[:, 2])  # enhance tumor
    
    loss_dict['ed_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['net_miss_dc_loss'] = dice_loss(batch_pred_missing[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_miss_dc_loss']  = dice_loss(batch_pred_missing[:, 2], batch_y[:, 2])  # enhance tumor

    ## Dice loss 
    loss_dict['loss_dc'] = loss_dict['ed_dc_loss'] + loss_dict['net_dc_loss'] + loss_dict['et_dc_loss']
    loss_dict['loss_miss_dc'] = loss_dict['ed_miss_dc_loss'] + loss_dict['net_miss_dc_loss'] + loss_dict['et_miss_dc_loss']
          
    ## Consistency loss
    loss_dict['ed_mse_loss']  = F.mse_loss(batch_pred_full[:, 0], batch_pred_missing[:, 0], reduction='mean') 
    loss_dict['net_mse_loss'] = F.mse_loss(batch_pred_full[:, 1], batch_pred_missing[:, 1], reduction='mean') 
    loss_dict['et_mse_loss']  = F.mse_loss(batch_pred_full[:, 2], batch_pred_missing[:, 2], reduction='mean') 
    loss_dict['consistency_loss'] = loss_dict['ed_mse_loss'] + loss_dict['net_mse_loss'] + loss_dict['et_mse_loss']
    
    weight_consistency = get_current_consistency_weight(epoch)
    
    return loss_dict['loss_dc'], loss_dict['loss_miss_dc'], weight_consistency * loss_dict['consistency_loss'] 


def simple_loss(batch_pred, batch_y):
    loss_dict = {}
    loss_dict['ed_dc_loss']  = dice_loss(batch_pred[:, 0], batch_y[:, 0])  # whole tumor
    loss_dict['net_dc_loss'] = dice_loss(batch_pred[:, 1], batch_y[:, 1])  # tumore core
    loss_dict['et_dc_loss']  = dice_loss(batch_pred[:, 2], batch_y[:, 2])  # enhance tumor
    loss = loss_dict['ed_dc_loss'] + loss_dict['net_dc_loss'] + loss_dict['et_dc_loss']
    return loss


def get_losses():
    losses = {}
    losses['co_loss'] = unet_Co_loss
    return losses


class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        prediction = torch.Tensor(prediction)
        target = torch.Tensor(target)
        iflat = prediction.reshape(-1)
        tflat = target.reshape(-1)
        intersection = (iflat * tflat).sum()

        return (2.0 * intersection + self.smooth) / (iflat.sum() + tflat.sum() + self.smooth)
    
    
class MS_SSIM_L1_LOSS(nn.Module):
    # from https://github.com/psyrocloud/MS-SSIM_L1_LOSS
    def __init__(self, gaussian_sigmas=[0.5, 1.0, 2.0, 4.0, 8.0],
                 data_range = 1.0,
                 K=(0.01, 0.03),
                 alpha=0.025,
                 compensation=200.0,
                 cuda_dev=0,):
        super(MS_SSIM_L1_LOSS, self).__init__()
        self.DR = data_range
        self.C1 = (K[0] * data_range) ** 2
        self.C2 = (K[1] * data_range) ** 2
        self.pad = int(2 * gaussian_sigmas[-1])
        self.alpha = alpha
        self.compensation=compensation
        filter_size = int(4 * gaussian_sigmas[-1] + 1)
        g_masks = torch.zeros((3*len(gaussian_sigmas), 1, filter_size, filter_size))
        for idx, sigma in enumerate(gaussian_sigmas):
            # r0,g0,b0,r1,g1,b1,...,rM,gM,bM
            g_masks[3*idx+0, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+1, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
            g_masks[3*idx+2, 0, :, :] = self._fspecial_gauss_2d(filter_size, sigma)
        self.g_masks = g_masks.cuda(cuda_dev)

    def _fspecial_gauss_1d(self, size, sigma):
        """Create 1-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 1D kernel (size)
        """
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        return g.reshape(-1)

    def _fspecial_gauss_2d(self, size, sigma):
        """Create 2-D gauss kernel
        Args:
            size (int): the size of gauss kernel
            sigma (float): sigma of normal distribution
        Returns:
            torch.Tensor: 2D kernel (size x size)
        """
        gaussian_vec = self._fspecial_gauss_1d(size, sigma)
        return torch.outer(gaussian_vec, gaussian_vec)

    def forward(self, x, y):
        b, c, h, w = x.shape
        mux = F.conv2d(x, self.g_masks, groups=1, padding=self.pad)
        muy = F.conv2d(y, self.g_masks, groups=1, padding=self.pad)

        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy

        sigmax2 = F.conv2d(x * x, self.g_masks, groups=1, padding=self.pad) - mux2
        sigmay2 = F.conv2d(y * y, self.g_masks, groups=1, padding=self.pad) - muy2
        sigmaxy = F.conv2d(x * y, self.g_masks, groups=1, padding=self.pad) - muxy

        # l(j), cs(j) in MS-SSIM
        l  = (2 * muxy    + self.C1) / (mux2    + muy2    + self.C1)  # [B, 15, H, W]
        cs = (2 * sigmaxy + self.C2) / (sigmax2 + sigmay2 + self.C2)

        lM = l[:, -1, :, :] * l[:, -2, :, :] * l[:, -3, :, :]
        PIcs = cs.prod(dim=1)

        loss_ms_ssim = 1 - lM*PIcs  # [B, H, W]

        loss_l1 = F.l1_loss(x, y, reduction='none')  # [B, 3, H, W]
        # average l1 loss in 3 channels
        gaussian_l1 = F.conv2d(loss_l1, self.g_masks.narrow(dim=0, start=-3, length=3),
                               groups=1, padding=self.pad).mean(1)  # [B, H, W]

        loss_mix = self.alpha * loss_ms_ssim + (1 - self.alpha) * gaussian_l1 / self.DR
        loss_mix = self.compensation*loss_mix

        return loss_mix.mean()