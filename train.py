import os
import sys
import logging
import datetime
import sys
from einops import rearrange
from einops.layers.torch import Rearrange
import argparse
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from model import MMCFormer
from dataset import make_data_loaders
from losses import get_losses, DiceLoss, MS_SSIM_L1_LOSS
from utils import measure_dice_score


use_cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser(description='MMCFormer')


parser.add_argument('--task_name',type=str, default='MMCFormer', 
                    help='task name')
parser.add_argument('--path_to_log', type=str, default='./new_results/', 
                    help='path to save results')
parser.add_argument('--path_to_data', type=str, default='../../brats/MICCAI_BraTS_2018_Data_Training/', 
                    help='path to dataset')

parser.add_argument('--context_loss', type=str, default='cosine', 
                    help='type = [cosine, L1]')
parser.add_argument('--context_loss_l1_coef', type=float, default=0.2, 
                    help='MCSA Loss: Level 1 Coef')
parser.add_argument('--context_loss_l2_coef', type=float, default=0.3, 
                    help='MCSA Loss: Level 2 Coef')
parser.add_argument('--context_loss_l3_coef', type=float, default=0.4, 
                    help='MCSA Loss: Level 3 Coef')
parser.add_argument('--context_loss_l4_coef', type=float, default=0.5, 
                    help='MCSA Loss: Level 4 Coef')
parser.add_argument('--context_loss_full_coef', type=float, default=0.01, 
                    help='MCSA Loss: All Level Coef')
parser.add_argument('--token_loss', type=str, default='L1', 
                    help='type = [L1, MSE]')
parser.add_argument('--token_loss_coef', type=float, default=0.005, 
                    help='MSP Loss: token loss Coef')
parser.add_argument('--recon_loss', type=str, default='L1', 
                    help='type = [MS-SSIM + L1, L1]')
parser.add_argument('--recon_loss_coef', type=float, default=0.01, 
                    help='Reconstruction Loss: loss Coef')
parser.add_argument('--consistency_coef', type=float, default=1.0, 
                    help='Consistency Loss: consistency loss Coef')
parser.add_argument('--weight_full_coef', type=float, default=0.4, 
                    help='Dice Loss: Coef for full modality')
parser.add_argument('--weight_missing_coef', type=float, default=0.6, 
                    help='Dice Loss: Coef for missing modality')

parser.add_argument('--modalities', type=str, nargs='*', default=['flair', 't1', 't1ce', 't2'], 
                    help='List of modalities needed to be used for training and evaluating the model (Sort [flair, t1, t1ce, t2] based on your desired modalities for the missing path)')
parser.add_argument('--n_missing_modalities', type=int, default=1, 
                    help='number of modalities for the missing path. WARNING: Sort [flair, t1, t1ce, t2] based on your desired modalities for the missing path')

parser.add_argument('--number_classes', type=int, default=4, 
                    help='number of classes in the target dataset')
parser.add_argument('--batch_size_tr', type=int, default=1, 
                    help='batch size for train')
parser.add_argument('--batch_size_va', type=int, default=1, 
                    help='batch size for validation')
parser.add_argument('--test_p', type=float, default=0.2, 
                    help='test percentage (20%)')
parser.add_argument('--progress_p', type=float, default=0.1, 
                    help='value between 0-1 shows the number of time we need to report training progress in each epoch')
parser.add_argument('--validation_p', type=float, default=0.1, 
                    help='validation percentage')
parser.add_argument('--inputshape', default=[128, 160, 192], 
                    help='input shape')

parser.add_argument('--n_epochs', type=int, default=200, 
                    help='number of epochs')
parser.add_argument('--lr', type=float,  default=1e-4,
                    help='segmentation network learning rate')
parser.add_argument('--weight_decay', type=float,  default=1e-5,
                    help='weight decay')
parser.add_argument('--power', type=float,  default=0.9,
                    help='power')

parser.add_argument('--missing_in_chans', type=int, default=1, 
                    help='missing modality input channels')
parser.add_argument('--full_in_chans', type=int, default=4, 
                    help='full modality input channels')


args = parser.parse_args()
task_name = args.task_name

#create log path
os.makedirs(args.path_to_log + task_name, exist_ok=True)  
log_dir = os.path.join(args.path_to_log, task_name)
    

# save logs
date_and_time = datetime.datetime.now()
logging.basicConfig(filename=args.path_to_log + task_name + f"/{task_name}" + str(date_and_time) + "_log.txt", level=logging.INFO,
                    format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
logging.info(f'{args}')


# load data
loaders = make_data_loaders(args)
for phase in ['train', 'eval']:
    loader = loaders[phase]
    total = len(loader)
    logging.info(f'Number of {phase} subjects: {total}')
    

def build_model(inp_shape, num_classes, full_in_chans, missing_in_chans):
    model_full    = MMCFormer(model_mode='full', img_size = inp_shape, num_classes=num_classes, in_chans=full_in_chans, 
                              head_count=1, token_mlp_mode="mix_skip").cuda()
    model_missing = MMCFormer(model_mode='missing', img_size = inp_shape, num_classes=num_classes, in_chans=missing_in_chans,
                              head_count=1, token_mlp_mode="mix_skip").cuda()
    
    return model_full, model_missing


def load_old_model(model_full, model_missing, optimizer, saved_model_path):
    print("Constructing model from saved file... ")
    checkpoint = torch.load(saved_model_path)
    model_full.load_state_dict(checkpoint["model_full"])
    model_missing.load_state_dict(checkpoint["model_missing"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epochs"]

    return model_full, model_missing, optimizer, epoch


class PolyLR(lr_scheduler._LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch. When last_epoch=-1, sets initial lr as lr.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self, optimizer, max_epoch, power=0.9, last_epoch=-1):
        self.max_epoch = max_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epoch) ** self.power
                for base_lr in self.base_lrs]

    
def make_optimizer_double(model1, model2):
    lr = args.lr
    optimizer = torch.optim.Adam([
    {'params': model1.parameters()},
    {'params': model2.parameters()}], lr=lr, weight_decay=args.weight_decay)
    scheduler = PolyLR(optimizer, max_epoch=args.n_epochs, power=args.power)

    return optimizer, scheduler


# Load Model
model_full, model_missing = build_model(inp_shape = args.inputshape, num_classes=args.number_classes,
                                        full_in_chans=args.full_in_chans, missing_in_chans=args.missing_in_chans)
optimizer, scheduler = make_optimizer_double(model_full, model_missing)


# Loss functions
losses = get_losses()
criteria = DiceLoss() 
mse_loss = torch.nn.MSELoss()
L1_loss = torch.nn.L1Loss()
L2_loss = torch.nn.MSELoss()
loss_cosine = nn.CosineEmbeddingLoss()
ms_ssim_l1_loss = MS_SSIM_L1_LOSS()

continue_training = False
epoch = 0
epoch_init = epoch


# training
n_epochs = args.n_epochs
iter_num = 0
best_dice = 0.0

for epoch in range(epoch_init, n_epochs):
    scheduler.step()
    
    train_loss = 0.0
    enc_loss_value = 0.0
    cls_loss_value = 0.0
    recon_loss_value = 0.0
    dice_full_loss_value = 0.0
    dice_missing_loss_value = 0.0
    consistency_loss_value = 0.0
    
    val_scores_full = 0.0
    val_scores_miss = 0.0
    val_loss_wt = 0.0
    val_loss_et = 0.0
    val_loss_ct = 0.0

    for phase in ['train', 'eval']:
        loader = loaders[phase]
        total = len(loader)
        for batch_id, (batch_x, batch_y) in enumerate(loader):
            iter_num = iter_num + 1
            batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
            batch_x = Rearrange('b c h w d -> b c d h w')(batch_x)
            batch_y = Rearrange('b c h w d -> b c d h w')(batch_y)
            
            with torch.set_grad_enabled(phase == 'train'):
                
                output_full, enc_context_att_full, CLS_full, recon_full = model_full(batch_x[:, 0:])  
#                 torch.cuda.clear_cache()
                output_missing, enc_context_att_missing, CLS_missing, _ = model_missing(batch_x[:, 0: args.n_missing_modalities])
                                
                ################# Loss functions #################
                ## MSCA Loss
                if args.context_loss == 'L1':
                    loss_context_att = (args.context_loss_l1_coef * L1_loss(enc_context_att_full[0], enc_context_att_missing[0]) + 
                                        args.context_loss_l2_coef * L1_loss(enc_context_att_full[1], enc_context_att_missing[1]) +
                                        args.context_loss_l3_coef * L1_loss(enc_context_att_full[2], enc_context_att_missing[2]) +
                                        args.context_loss_l4_coef * L1_loss(enc_context_att_full[3], enc_context_att_missing[3]))
                else:
                    enc_context_att_full[0] = enc_context_att_full[0].reshape(1, -1)
                    loss_context_att = (args.context_loss_l1_coef * loss_cosine(enc_context_att_full[0].reshape(1, -1), enc_context_att_missing[0].reshape(1, -1), torch.tensor([1]).cuda()) + 
                                        args.context_loss_l2_coef * loss_cosine(enc_context_att_full[1].reshape(1, -1), enc_context_att_missing[1].reshape(1, -1), torch.tensor([1]).cuda()) +
                                        args.context_loss_l3_coef * loss_cosine(enc_context_att_full[2].reshape(1, -1), enc_context_att_missing[2].reshape(1, -1), torch.tensor([1]).cuda()) +
                                        args.context_loss_l4_coef * loss_cosine(enc_context_att_full[3].reshape(1, -1), enc_context_att_missing[3].reshape(1, -1), torch.tensor([1]).cuda()))
                
                
                ## Dice and Consistency losses
                loss_dc, loss_miss_dc, consistency_loss = losses['co_loss'](output_full, output_missing, batch_y, epoch)


                ## MSP Loss
                if args.token_loss == 'L1':
                    cls_token_loss = L1_loss(CLS_full, CLS_missing)
                else:
                    cls_token_loss = L2_loss(CLS_full, CLS_missing) 
                
                
                ## Reconstruction loss
                if args.recon_loss == 'L1':
                    recon_loss = L1_loss(recon_full, batch_x[:, 0:])
                else:
                    recon_full_prime = recon_full.permute(0,2,1,3,4)[0]
                    batch_x_prime = batch_x.permute(0,2,1,3,4)[0]
                    recon_loss = (ms_ssim_l1_loss(recon_full_prime[:,0:1,...], batch_x_prime[:,0:1,...]) +
                                  ms_ssim_l1_loss(recon_full_prime[:,1:2,...], batch_x_prime[:,1:2,...]) +
                                  ms_ssim_l1_loss(recon_full_prime[:,2:3,...], batch_x_prime[:,2:3,...]) +
                                  ms_ssim_l1_loss(recon_full_prime[:,3:4,...], batch_x_prime[:,3:4,...]))                
                
                # Total loss
                tot_loss = args.weight_full_coef * loss_dc + \
                         args.weight_missing_coef * loss_miss_dc + \
                         args.consistency_coef * consistency_loss + \
                         args.context_loss_full_coef * loss_context_att + \
                         args.token_loss_coef * cls_token_loss + \
                         args.recon_loss_coef * recon_loss
            
                optimizer.zero_grad()

                if phase == 'train':
                    tot_loss.backward(retain_graph=True)
                    train_loss += tot_loss.item()
                    enc_loss_value += (args.context_loss_full_coef * loss_context_att.item())
                    dice_full_loss_value += (args.weight_full_coef * loss_dc.item())
                    dice_missing_loss_value += (args.weight_missing_coef * loss_miss_dc.item())
                    consistency_loss_value += (args.consistency_coef * consistency_loss.item())
                    cls_loss_value += (args.token_loss_coef * cls_token_loss.item())
                    recon_loss_value += (args.recon_loss_coef * recon_loss.item())

            if phase == 'train':
                optimizer.step()
                if (batch_id + 1) % 20 == 0:
                    logging.info(f'Epoch: {epoch+1}|| iteration: {batch_id+1}|| Training loss: {(train_loss/(batch_id+1)):.5f}|| Encoder loss: {(enc_loss_value/(batch_id+1)):.7f}|| DSC loss full: {(dice_full_loss_value/(batch_id+1)):.5f}|| DSC loss missing: {(dice_missing_loss_value/(batch_id+1)):.5f}|| Consistency loss: {(consistency_loss_value/(batch_id+1)):.5f}|| CLS loss: {(cls_loss_value/(batch_id+1)):.6f}|| Recon loss: {(recon_loss_value/(batch_id+1)):.7f}')                                
            else:
                val_scores_full_t, val_loss_full_wt, val_loss_full_et, val_loss_full_ct = measure_dice_score(output_full, batch_y)
                val_scores_miss_t, val_loss_missing_wt_t, val_loss_missing_et_t, val_loss_missing_ct_t = measure_dice_score(output_missing, batch_y)
                
                val_scores_full += val_scores_full_t
                val_scores_miss += val_scores_miss_t
                val_loss_wt += val_loss_missing_wt_t
                val_loss_et += val_loss_missing_et_t
                val_loss_ct += val_loss_missing_ct_t
                
            
        if phase == 'train':
            logging.info(f'### Epoch {epoch+1} overall training loss>> {train_loss/(batch_id+1)}')
        else:
            dice_full = (val_scores_full/(batch_id+1))
            dice_missing = (val_scores_miss/(batch_id+1)) 
            dice_wt = (val_loss_wt/(batch_id+1))
            dice_et = (val_loss_et/(batch_id+1))
            dice_ct = (val_loss_ct/(batch_id+1))
            

            # save model
            state = {}
            state['model_full'] = model_full.state_dict()
            state['model_missing'] = model_missing.state_dict()
            state['optimizer'] = optimizer.state_dict()
            state['epochs'] = epoch
            file_name = log_dir + '/model_weights.pth'
            torch.save(state, file_name)
            
            if dice_missing > best_dice:
                torch.save(state, log_dir + '/best_model_weights.pth')
                best_dice = dice_missing 
            logging.info(f'### Best Val DSC missing >> {best_dice}')
            logging.info(f'### Epoch {epoch+1}, Val DSC full: {dice_full}, Val DSC missing: {dice_missing}, WT: {dice_wt}, CT: {dice_ct}, ET: {dice_et}')
            