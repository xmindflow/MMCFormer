import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=3, embed_dim=96, norm_layer=None, 
                stride=4, padding=1):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        input:  B, C, D, H, W
        Output: B, D, H, W, C
        """
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.view(-1, D, Wh, Ww, self.embed_dim)
            _, D, H, W, C = x.shape

        return x, D, H, W
    
    
class EfficientAttention3D(nn.Module):
    """
        input  -> x:[B, C, D, H, W]
    
        in_channels:    int -> Embedding Dimension 
        key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
        value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2) 
        head_count:     int -> It divides the embedding dimension by the head_count and process each part individually
    """
    
    def __init__(self, in_channels, key_channels, value_channels, head_count=1, recon_mode=False):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.recon_mode = recon_mode
            
        self.keys = nn.Conv3d(in_channels, key_channels, 1) 
        self.queries = nn.Conv3d(in_channels, key_channels, 1)
        self.values = nn.Conv3d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv3d(value_channels, in_channels, 1)
    
        
    def forward(self, input_, CLS):
        n, c, d, h, w = input_.size()
                
        keys = self.keys(input_).reshape((n, self.key_channels, d * h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, d * h * w)
        values = self.values(input_).reshape((n, self.value_channels, d * h * w))
            
        if CLS is not None:
            keys = torch.cat((CLS, keys), dim=-1)
            queries = torch.cat((CLS, queries), dim=-1)
            values = torch.cat((CLS, values), dim=-1)
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
            
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]            
            
            context = key @ value.transpose(1, 2) # dk*dv
            attended_value = (context.transpose(1, 2) @ query)
                        
            if CLS is not None:
                CLS, attended_value = attended_value[..., :4], attended_value[..., 4:]

            attended_value = attended_value.reshape(n, head_value_channels, d, h, w) # n*dv 
            attended_values.append(attended_value)
                
        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)
        
        if CLS is not None: 
            return attention, context, CLS
        return attention, context
    
    
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, D, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, D, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
    def forward(self, x: torch.Tensor, D, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), D, H, W)+self.fc1(x)))
        out = self.fc2(ax)
        return out

    
class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: torch.Tensor, D, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), D, H, W))
        out = self.fc2(ax)
        return out

    
class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x, D, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
    
class EfficientTransformerBlock3D(nn.Module):
    """
    input:  B, D, H, W, C
    Output: B, D, H, W, C
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp='mix', recon_mode=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention3D(in_channels=in_dim, key_channels=key_dim,
                                       value_channels=value_dim, head_count=1, recon_mode=recon_mode)
        self.norm2 = nn.LayerNorm(in_dim)
        if token_mlp=='mix':
            self.mlp = MixFFN(in_dim, int(in_dim*4))  
        elif token_mlp=='mix_skip':
            self.mlp = MixFFN_skip(in_dim, int(in_dim*4)) 
        else:
            self.mlp = MLP_FFN(in_dim, int(in_dim*4))

    def forward(self, x: torch.Tensor, D, H, W, CLS=None) -> torch.Tensor:
        norm_1 = self.norm1(x)
        norm_1 = Rearrange('b d h w c -> b c d h w', d=D, h=H, w=W)(norm_1)
        
        if CLS is not None:
            attn, context, CLS = self.attn(norm_1, CLS=CLS)
        else: 
            attn, context= self.attn(norm_1, CLS=CLS)
        attn = Rearrange('b c d h w -> b d h w c')(attn)
        
        tx = x + attn
        tx = Rearrange('b d h w c -> b (d h w) c')(tx)

        mx = tx + self.mlp(self.norm2(tx), D, H, W)
        mx = Rearrange('b (d h w) c -> b d h w c', d=D, h=H, w=W)(mx)
        
        return mx, context, CLS
    

############################################## Encoder ##############################################  
class Encoder(nn.Module):
    def __init__(self, img_size, in_dim, key_dim, value_dim, layers, patch_sizes, in_chans=4,
                 norm_layer=nn.LayerNorm, patch_norm=True, head_count=1, token_mlp='mix_skip'):
        super().__init__()
        
        strides = [(4,4,4), (2,2,2), (2,2,2)]
        padding = [(0,0,0), (0,0,1), (0,0,1)]
        
        # patch_embed
        # layers = [2, 2, 2] dims = [64, 128, 256]
        self.patch_embed1 = PatchEmbed3D(img_size=img_size, patch_size=patch_sizes[0], in_chans=in_chans,
                                        embed_dim=in_dim[0], norm_layer=norm_layer if patch_norm else None,
                                        stride=strides[0], padding=padding[0])
        self.patch_embed2 = PatchEmbed3D(img_size=np.floor_divide(img_size, 4), patch_size=patch_sizes[1], in_chans=in_dim[0],
                                        embed_dim=in_dim[1], norm_layer=norm_layer if patch_norm else None,
                                        stride=strides[1],padding=padding[1])        
        self.patch_embed3 = PatchEmbed3D(img_size=np.floor_divide(img_size, 8), patch_size=patch_sizes[2], in_chans=in_dim[1],
                                        embed_dim=in_dim[2], norm_layer=norm_layer if patch_norm else None,
                                        stride=strides[2],padding=padding[2])
        
        
        # transformer encoder
        self.block1 = nn.ModuleList([ 
            EfficientTransformerBlock3D(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp)
        for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList([
            EfficientTransformerBlock3D(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp)
        for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList([
            EfficientTransformerBlock3D(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp)
        for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(in_dim[2])

        # Define class token
        self.cls_token = nn.Parameter(torch.zeros(1, in_dim[2], 4))
        trunc_normal_(self.cls_token, std=.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []
        context_att = []
        
        # stage 1
        x, D, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x, context, _ = blk(x, D, H, W)
            context_att.append(context)
        x = self.norm1(x)
        outs.append(x)
        

        # stage 2
        x = Rearrange('b d h w c -> b c d h w')(x)
        x, D, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x, context, _ = blk(x, D, H, W)
            context_att.append(context)
        x = self.norm2(x)
        outs.append(x)
        
        
        # stage 3
        x = Rearrange('b d h w c -> b c d h w')(x)
        x, D, H, W = self.patch_embed3(x)
        
        # token loss
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        for blk in self.block3:
            x, context, cls_tokens = blk(x, D, H, W, cls_tokens)
        x = self.norm3(x)
        x = Rearrange('b d h w c -> b (d h w) c')(x)
        outs.append(x)

        return outs, context_att, cls_tokens
    

############################################## Decoder ##############################################
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim_scale = dim_scale
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        D, H, W = self.input_resolution
        x = x.flatten(2)
        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale,
                      p3=self.dim_scale, c=C // 8)
        x = self.norm(x)

        return x
    

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 4 * 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, D*H*W, C
        """
        D, H, W = self.input_resolution

        x = self.expand(x)
        B, L, C = x.shape

        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 p3 c)-> b (d p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale,
                      p3=self.dim_scale,
                      c=C // (self.dim_scale ** 3))
        
        x = self.norm(x)

        return x
    

class MyDecoderLayer(nn.Module):
    def __init__(self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9,
                 norm_layer=nn.LayerNorm, is_last=False, recon_mode=False):
        super().__init__()
        
        self.recon_mode = recon_mode
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        
        self.is_last = is_last
        
        if not is_last:
            self.concat_linear = nn.Linear(dims*2, out_dim)
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.concat_linear = nn.Linear(dims*4, out_dim)
            if recon_mode:
                self.layer_up = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
            else:
                self.layer_up = FinalPatchExpand_X4(input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            self.last_layer = nn.Conv3d(out_dim, n_class, 1, bias=False)
        
        if self.recon_mode == False:
            self.layer_former_1 = EfficientTransformerBlock3D(out_dim, key_dim, value_dim, head_count, 
                                                              token_mlp_mode, recon_mode=recon_mode)
        self.layer_former_2 = EfficientTransformerBlock3D(out_dim, key_dim, value_dim, head_count, 
                                                          token_mlp_mode, recon_mode=recon_mode)
            
        
        def init_weights(self): 
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)
      
    def forward(self, x1, x2=None, first=False, CLS=None):
        
        if first:
            out = self.layer_up(x1)
        else:
            b, d, h, w, c = x2.shape
            
            
            cat_x = torch.cat([x1, x2], dim=-1)
            cat_x = cat_x.view(b, -1, cat_x.shape[-1])
            cat_linear_x = self.concat_linear(cat_x)
            cat_linear_x = Rearrange('b (d h w) c -> b d h w c', b=b, h=h, w=w)(cat_linear_x)
            
            if self.recon_mode == False:
                cat_linear_x, _, _ = self.layer_former_1(cat_linear_x, d, h, w, CLS=CLS)
            tran_layer_2, _, _ = self.layer_former_2(cat_linear_x, d, h, w, CLS=CLS)                
            tran_layer_2 = Rearrange('b d h w c -> b (d h w) c')(tran_layer_2)
            
            
            if self.is_last:
                if self.recon_mode:
                    tran_layer_2 = Rearrange('b (d h w) c -> b c d h w', b=b, h=h, w=w)(tran_layer_2)
                out = self.layer_up(tran_layer_2)
                if not self.recon_mode:
                    out = Rearrange('b d h w c -> b c d h w')(out)
                out = self.last_layer(out)  
            else:
                out = self.layer_up(tran_layer_2)  
                
        return out
    

############################################## MMCFormer ##############################################    
class MMCFormer(nn.Module):
    def __init__(self, model_mode, img_size = (128, 160, 192), num_classes=4, in_chans=4, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder  
        in_dim, key_dim, value_dim, layers =[[64, 128, 320], [64, 128, 320], [64, 128, 320], [2, 2, 2]] 
        patch_sizes = [(4, 4, 4), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        
        self.enc = Encoder(img_size, in_dim, key_dim, value_dim, layers, patch_sizes,
                          in_chans = in_chans, head_count=1, token_mlp='mix_skip')
        
        # Decoder
        d_base_feat_size = [4, 5, 6]
        in_out_chan = [[32, 64, 64, 64],[144, 128, 128, 128],[288, 320, 320, 320]]   #[dim, out_dim, key_dim, value_dim]
        
        self.decoder_2 = MyDecoderLayer((d_base_feat_size[0]*2, d_base_feat_size[1]*2, d_base_feat_size[2]*2),
                                        in_out_chan[2], head_count, token_mlp_mode, n_class=num_classes, recon_mode=False)

        self.decoder_1 = MyDecoderLayer((d_base_feat_size[0]*4, d_base_feat_size[1]*4, d_base_feat_size[2]*4),
                                        in_out_chan[1], head_count, token_mlp_mode, n_class=num_classes, recon_mode=False)

        self.decoder_0 = MyDecoderLayer((d_base_feat_size[0]*8, d_base_feat_size[1]*8, d_base_feat_size[2]*8),
                                        in_out_chan[0], head_count, token_mlp_mode, n_class=num_classes,
                                        is_last=True, recon_mode=False) 


        self.model_mode = model_mode
        if self.model_mode == 'full':
            self.decoder_recon = MyDecoderLayer((d_base_feat_size[0]*8, d_base_feat_size[1]*8, d_base_feat_size[2]*8),
                                                in_out_chan[0], head_count, token_mlp_mode, n_class=num_classes,
                                                is_last=True, recon_mode=True)
            self.cls_projection = nn.Linear(in_out_chan[2][-1], in_out_chan[0][-1])
        
    def forward(self, x):
        """
        input:  B, D*H*W, C
        output: B, C, D, H, W
        
        """
        
        enc_out, enc_context_att, CLS = self.enc(x)
        CLS = CLS.permute(0, 2, 1)
                
    
        # stage2 
        tmp_2 = self.decoder_2(enc_out[2], first=True) # B,D,H,W,C

        # stage1   
        tmp_1 = self.decoder_1(tmp_2, enc_out[1], first=False) # B,D,H,W,C

        # stage0
        tmp_seg = self.decoder_0(tmp_1, enc_out[0], first=False) # B,D,H,W,C
        uout = torch.sigmoid(tmp_seg)
        
        # Recon stage
        if self.model_mode == 'full':
            proj_CLS = self.cls_projection(CLS).permute(0, 2, 1)
            tmp_recon = self.decoder_recon(tmp_1, enc_out[0], first=False, CLS=proj_CLS)
            
            return uout, enc_context_att, CLS, tmp_recon

        return uout, enc_context_att, CLS, []