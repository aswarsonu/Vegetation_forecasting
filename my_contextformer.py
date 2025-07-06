import argparse
import ast
from typing import Optional, Union

import numpy as np
import timm
import torch
from torch.cuda.amp import autocast    
import torch.nn as nn
from earthnet_models_pytorch.model.layer_utils import inverse_permutation
from earthnet_models_pytorch.utils import str2bool
from torch.jit import Final
from torch.nn import functional as F
from dataclasses import dataclass
from torchvision.transforms.functional import to_pil_image

import argparse
import ast
from transformers import CLIPModel, CLIPConfig, CLIPProcessor
from typing import List, Optional, Union
from transformers import CLIPModel, CLIPProcessor
import argparse
import numpy as np
import safetensors.torch
import timm
import torch
import torch.nn as nn
from earthnet_models_pytorch.model.layer_utils import inverse_permutation
from earthnet_models_pytorch.utils import str2bool
from torch.jit import Final
from torch.nn import functional as F
import math
import os
from pathlib import Path
# itransformer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from transformers import AutoModel, AutoTokenizer
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import torch
import torch.nn as nn
import safetensors.torch

from transformers import (
    CLIPProcessor,
    CLIPModel,
    CLIPConfig,
)


class Attention(nn.Module):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x





#### ─────────────────────────── CLIP Block ──────────────────────────────────── #####


class CLIPPatchEmbed(nn.Module):
    """
    Encode full-size frames with CLIP vision, then project to hidden_dim.
    (keeps class name and self.proj as before)
    """

    def __init__(
        self,
        clip_checkpoint: str | None = None,
        freeze: bool = True,
        patch_size: int = 4,         # still kept, but unused now
        hidden_dim: int = 128
    ):
        super().__init__()
        self.patch_size = patch_size

        ckpt = Path(clip_checkpoint) if clip_checkpoint else CLIP_PATH
        try:
            # pull out just the vision tower
            self.clip = CLIPModel.from_pretrained(str(ckpt)).vision_model
        except Exception:
            cfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
            self.clip = CLIPModel(cfg).vision_model
            # …you can inject your safetensors weights here if needed…

        if freeze:
            self.clip.eval()
            for p in self.clip.parameters():
                p.requires_grad = False

        # same projection as before
        self.proj = nn.Linear(self.clip.config.hidden_size, hidden_dim)

    @torch.no_grad()
    def forward(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        """
        patch_tensor: now [B, T, 3, H, W]
        returns     : [B, T, hidden_dim]
        """
        B, T, C, H, W = patch_tensor.shape
        device = next(self.clip.parameters()).device

        # flatten batch+time
        x = patch_tensor.view(B * T, C, H, W).to(device).float()

        # scale to [0,1]
        if x.max() > 1.0:
            x = x.div_(255.0)

        # normalize with CLIP’s ImageNet stats
        # mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
        #                    device=device).view(1, 3, 1, 1)
        #std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
        #                    device=device).view(1, 3, 1, 1)
        #x = (x - mean) / std

        # get CLIP image features → [B*T, hidden_size]
        feats = self.clip.get_image_features(pixel_values=x)

        # reshape back → [B, T, hidden_size]
        emb = feats.view(B, T, -1)

        # final projection
        return self.proj(emb)  # [B, T, hidden_dim]
               




# ------------------------------------------------------------------
#  Default location of your fine‑tuned checkpoint (can be overriden
#  by the ctor arg or by exporting CLIP_CHECKPOINT=/path/to/ckpt)
# ------------------------------------------------------------------




class CLIPTextEncoder(nn.Module):
    """
    Lightweight wrapper around CLIP’s text encoder.
    Loads either a Hugging‑Face‑style folder (config + weights) or a bare
    weights file under `clip_checkpoint`.  Exposes `self.processor`
    and `self.clip` so that upstream models can reuse them.
    """

    def __init__(self,
                 clip_checkpoint: str | None = None,
                 freeze: bool = True):
        super().__init__()

        # ①  The tokenizer / pre‑processor ---------------------------------
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # ②  Locate checkpoint folder --------------------------------------
        ckpt_dir = Path(clip_checkpoint) if clip_checkpoint else CLIP_PATH

        # ③  Try the “proper” HF repo first --------------------------------
        try:
            self.clip = CLIPModel.from_pretrained(str(ckpt_dir))
        except Exception:                       # no config.json, etc.
            # build a base config …
            cfg = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
            self.clip = CLIPModel(cfg)

            # … then inject weights (prefer safetensors)
            if (ckpt_dir / "model.safetensors").is_file():
                state = safetensors.torch.load_file(str(ckpt_dir / "model.safetensors"))
            else:
                state = torch.load(str(ckpt_dir / "pytorch_model.bin"),
                                   map_location="cpu")

            # strip any “inner.” prefixes your fine‑tuner might have added
            clean_state = {k.removeprefix("inner."): v for k, v in state.items()}

            # load (strict=False lets us ignore non‑matching keys)
            self.clip.load_state_dict(clean_state, strict=False)

        # ④  Optionally freeze ---------------------------------------------
        if freeze:
            self.clip.eval()
            for p in self.clip.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self,
                captions: list[str],
                device: torch.device) -> torch.Tensor:
        """
        Args
        ----
        captions : list[str]        length = B
        device   : torch.device     where the tensors should live

        Returns
        -------
        torch.Tensor shape (B, 512)
        """
        batch = self.processor(
            text=captions,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        return self.clip.get_text_features(**batch)


#### ─────────────────────────── I-Transformer Block ──────────────────────────────────── #####
class ITransformerBackbone(nn.Module):
    """
    Input  : tok  [B_patch, T, D]
    Output : same shape   (ready for self.head)
    Internally: LayerNorm + sine-pos-enc  + standard iTransformer Encoder
    """
    def __init__(self, d_model, n_heads, depth, d_ff, dropout, factor=5):
        super().__init__()
        # plain LayerNorm instead of DataEmbedding_inverted (keeps D unchanged)
        self.pre_norm = nn.LayerNorm(d_model)

        # sine/cosine positional enc. identical to old get_sinusoid_encoding_table
        self.register_buffer(
            "pos_enc",
            get_sinusoid_encoding_table(2048, d_model),   # 2048≫T upper-bound
            persistent=False,
        )

        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, factor,
                                  attention_dropout=dropout,
                                  output_attention=False),
                    d_model, n_heads
                ),
                d_model, d_ff,
                dropout=dropout,
                activation="gelu",
            )
            for _ in range(depth)
        ], norm_layer=nn.LayerNorm(d_model))

    def forward(self, tok: torch.Tensor) -> torch.Tensor:          # [B_p,T,D]
        b, t, d = tok.shape
        x = self.pre_norm(tok) + self.pos_enc[:t].unsqueeze(0)     # add pos
        x, _ = self.encoder(x, attn_mask=None)                     # iT blocks
        return x                                                   # [B_p,T,D]

class Model(nn.Module):
    """
    Transformer-based Time Series Forecasting Model. (itransformer block)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # 10 (Input sequence length)
        self.pred_len = configs.pred_len  # 20 (Prediction length)
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding Layer
        self.enc_embedding = DataEmbedding_inverted(
            c_in=configs.c_in,   # 8 weather features
            d_model=configs.d_model,
            embed_type=configs.embed_type,
            freq=configs.freq,
            dropout=configs.dropout
        )

        # Encoder Layer
        self.encoder = Encoder([
            EncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, 
                                  attention_dropout=configs.dropout,
                                  output_attention=configs.output_attention),
                    configs.d_model, configs.n_heads
                ),
                configs.d_model, configs.d_ff, dropout=configs.dropout,
                activation=configs.activation
            ) for _ in range(configs.e_layers)
        ], norm_layer=torch.nn.LayerNorm(configs.d_model))

        # Linear Projection Layer for prediction
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forward(self, x_enc):
        return self.forecast(x_enc)

    def forecast(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding
        enc_out = self.enc_embedding(x_enc)  # [B, L, d_model]

        # Encoder
        enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [B, L, d_model]

        # Project to prediction
        dec_out = self.projector(enc_out[:, -1, :])  # Take the last output for prediction [B, pred_len]
        dec_out = dec_out.unsqueeze(2).repeat(1, 1, x_enc.size(2))  # [B, pred_len, num_features]

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out  # [B, pred_len, num_features]

#### ─────────────────────────── code for Contextformer (baseline code) ──────────────────────────────────── #####

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
        
    

    


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table)

#### ─────────────────────────── Code from baseline ──────────────────────────────────── #####
class PVT_embed(nn.Module):

    def __init__(self, in_channels, out_channels, pretrained=True, frozen=False):
        super().__init__()

        self.pvt = timm.create_model(
            "pvt_v2_b0.in1k",
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
        )
        if frozen:
            timm.utils.freeze(self.pvt)
        self.pvt_project = nn.Conv2d(
            in_channels=512,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=True,
        )

    def forward(self, x):

        B, T, C, H, W = x.shape

        x_feats = self.pvt(x.reshape(B * T, C, H, W))

        x_feats = [F.interpolate(feat, size=x_feats[0].shape[-2:]) for feat in x_feats]

        x = self.pvt_project(torch.cat(x_feats, dim=1))

        _, C, H, W = x.shape

        # Patchify

        x_patches = (
            x.reshape(B, T, C, H, W).permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        )

        return x_patches




class ContextFormer(nn.Module):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
    
    
#### ─────────────────────────── Creating CLIP Image and Text Encoder───────────────────────────────────── #####
            
        self.img_patch = CLIPPatchEmbed(
            clip_checkpoint=hparams.clip_checkpoint,
            freeze=hparams.freeze_clip,
            patch_size=hparams.patch_size,
            hidden_dim=hparams.n_hidden,        
        )
        
        self.txt_enc   = CLIPTextEncoder(
            clip_checkpoint=hparams.clip_checkpoint,
            freeze=hparams.freeze_clip,
        )
        self.txt_proj  = nn.Linear(512, hparams.n_hidden)
        self.txt_norm = nn.LayerNorm(hparams.n_hidden) 
        self.text_scale = nn.Parameter(torch.tensor(0.5))
        if self.hparams.pvt:
            self.embed_images = PVT_embed(
                in_channels=self.hparams.n_image,
                out_channels=self.hparams.n_hidden,
                pretrained=True,
                frozen=self.hparams.pvt_frozen,
            )
        else:
            self.embed_images = self.img_patch

        self.embed_weather = Mlp(
            in_features=self.hparams.n_weather,
            hidden_features=self.hparams.n_hidden,
            out_features=self.hparams.n_hidden,
        )

        self.mask_token = nn.Parameter(torch.zeros(self.hparams.n_hidden))

        self.blocks = nn.ModuleList(
            [
                Block(
                    self.hparams.n_hidden,
                    self.hparams.n_heads,
                    self.hparams.mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(self.hparams.depth)
            ]
        )

        
        self.head = Mlp(
            in_features=self.hparams.n_hidden,
            hidden_features=self.hparams.n_hidden,
            out_features=self.hparams.n_out
            * self.hparams.patch_size
            * self.hparams.patch_size,
        )
        # this MLP turns the latent back into pixel-space patches
        self.delta_head = Mlp(
            in_features    = self.hparams.n_hidden,
            hidden_features= self.hparams.n_hidden,
            out_features   = self.hparams.n_out
                        * self.hparams.patch_size
                        * self.hparams.patch_size,
        )
        
        if self.hparams.predict_delta_avg or self.hparams.predict_delta_max:
            self.delta_head_avg = Mlp(
                in_features    = self.hparams.n_hidden,
                hidden_features= self.hparams.n_hidden,
                out_features   = self.hparams.n_out
                            * self.hparams.patch_size
                            * self.hparams.patch_size,
            )

        

        
        

    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--setting", type=str, default="en21x")
        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)
        parser.add_argument("--patch_size", type=int, default=8)
        parser.add_argument("--n_image", type=int, default=8)
        parser.add_argument("--n_weather", type=int, default=24)
        parser.add_argument("--n_hidden", type=int, default=256)
        parser.add_argument("--n_out", type=int, default=1)
        parser.add_argument("--n_heads", type=int, default=8)
        parser.add_argument("--depth", type=int, default=3)
        parser.add_argument("--mlp_ratio", type=float, default=4.0)
        parser.add_argument("--mtm", type=str2bool, default=False)
        parser.add_argument("--leave_n_first", type=int, default=3)
        parser.add_argument("--p_mtm", type=float, default=0.3)
        parser.add_argument("--p_use_mtm", type=float, default=1.0)
        parser.add_argument("--mask_clouds", type=str2bool, default=False)
        parser.add_argument("--use_weather", type=str2bool, default=True)
        parser.add_argument("--predict_delta", type=str2bool, default=False)
        parser.add_argument("--predict_delta0", type=str2bool, default=False)
        parser.add_argument("--predict_delta_avg", type=str2bool, default=False)
        parser.add_argument("--predict_delta_max", type=str2bool, default=False)
        parser.add_argument("--pvt", type=str2bool, default=False)
        parser.add_argument("--pvt_frozen", type=str2bool, default=False)
        parser.add_argument("--add_last_ndvi", type=str2bool, default=False)
        parser.add_argument("--add_mean_ndvi", type=str2bool, default=False)
        parser.add_argument("--spatial_shuffle", type=str2bool, default=False)
        parser.add_argument("--freeze_clip", type=lambda x: x.lower() in ("1", "true", "yes"), default=True, help="whether to freeze the CLIP encoder weights" )
        parser.add_argument("--dropout", type=float, default=0.1,
                   help="dropout rate for all iTransformer blocks")       
        parser.add_argument("--clip_checkpoint", type=str, default="/storage/ij_sonu/checkpoint-169410", help="Path to your fine-tuned CLIP checkpoint folder")
        parser.add_argument("--use_itransformer",type=str2bool, default=False)
        return parser

    def forward(self, data, pred_start: int = 0, preds_length: Optional[int] = None):

        
#### ─────────────────────────── Input Handling (Same from baseline)───────────────────────────────────── #####
        
        

        device        = self.mask_token.device
        c_l           = self.hparams.context_length if self.training else pred_start
        preds_length  = preds_length or 0  

        preds_length = preds_length
        
       
       
        
        # unpack raw tensors --------------------------------------------------
        hr_dynamic_inputs = data["dynamic"][0].to(device)     # [B,T,C,H,W]
        hr_dynamic_mask   = data["dynamic_mask"][0].to(device) # [B,T,1,H,W]
        static_inputs     = data["static"][0][:, :3, ...].to(device)           # [B,3,H,W]
        weather           = data["dynamic"][1].to(device)       # [B,10,24]

        B, T, C, H, W = hr_dynamic_inputs.shape

        # stitch dynamic + static → full image stack -------------------------
        images = torch.cat(
            [hr_dynamic_inputs,
                static_inputs.unsqueeze(1).repeat(1, T, 1, 1, 1)],
            dim=2
        )    
        # patch-level cloud mask ------------------------------------------------
        mask_patches = (
            hr_dynamic_mask.reshape(
                B, T, 1,
                H // self.hparams.patch_size, self.hparams.patch_size,
                W // self.hparams.patch_size, self.hparams.patch_size
            )
            .permute(0, 3, 5, 1, 2, 4, 6)
            .reshape(
                B * H // self.hparams.patch_size * W // self.hparams.patch_size,
                T,
                self.hparams.patch_size ** 2
            )
        )

#### ─────────────────────────── embed image patches (PVT or CLIP-patch)───────────────────────────────────── #####
        
        if self.hparams.pvt:
            image_patches_embed = self.embed_images(images)          # [B_p,T,H]
        else:
                        # full-frame CLIP embedding per timestep, then broadcast to patches
            B, T, C, H, W = images.shape
            patches_per_img = (H // self.hparams.patch_size) * (W // self.hparams.patch_size)

            # flatten into (B*T, C, H, W)
            frames = images.view(B * T, C, H, W).to(device)
            # ensure 3-ch
            if frames.shape[1] == 1:
                frames = frames.repeat(1, 3, 1, 1)
            elif frames.shape[1] > 3:
                frames = frames[:, :3]

            # upsample to CLIP’s 224×224 and normalize
            frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
            if frames.dtype == torch.uint8:
                frames = frames.float().div_(255)
            elif frames.max() > 1.5:
                frames = frames.float().div_(255)
            frames = (frames - _CLIP_MEAN.to(device)) / _CLIP_STD.to(device)
            frames = frames.to(dtype=next(self.img_patch.clip.parameters()).dtype)

            
            feats = []
            for chunk in torch.split(frames, 64, dim=0):
                feats.append(self.img_patch.clip.get_image_features(pixel_values=chunk))
            feats = torch.cat(feats, 0)                   # [B*T, 512]
            feats = feats.to(self.img_patch.proj.weight.dtype)
            feats = self.img_patch.proj(feats)            # [B*T, hidden]

            # reshape back to [B, T, hidden]
            emb = feats.view(B, T, -1)

            
            image_patches_embed = emb.unsqueeze(1)                         # [B, 1, T, H]
            image_patches_embed = image_patches_embed.expand(
                B, patches_per_img, T, emb.shape[2]
            ).reshape(B * patches_per_img, T, emb.shape[2])     

        B_patch, N_patch, _ = image_patches_embed.shape
        patches_per_img     = (H // self.hparams.patch_size) * (W // self.hparams.patch_size)
        B                   = B_patch // patches_per_img

        # caption Embeddings

#### ─────────────────────────── caption processing (CLIP)───────────────────────────────────── #####
                
        raw_caps = data.get("llama_caption", [[""] * T for _ in range(B)])
        caps = []
        for b in range(B):
            caps_b = raw_caps[b]
            if len(caps_b) < T:
                caps_b += [""] * (T - len(caps_b))
            else:
                caps_b = caps_b[:T]
            caps.extend(caps_b)
        cap_emb = self.txt_enc(caps, device)           # [(B*T),512]
        cap_emb = self.txt_norm(self.txt_proj(cap_emb)).mul_(self.text_scale)
        cap_emb = cap_emb.view(B, T, -1)               # [B,T,H]
        cap_patch = cap_emb.unsqueeze(1).expand(-1, patches_per_img, -1, -1)\
                                    .reshape(B_patch, T, -1)



        # getting weather patches
        weather_patches = (
            weather.unsqueeze(1)                        # [B,1,30,24]
                    .repeat(1, patches_per_img, 1, 1)     # [B,p,30,24]
                    .reshape(B_patch, 30, 24)
        )
        weather_patches_embed = self.embed_weather(weather_patches) # [B_p,T,H]
                                        # [B,T,C+3,H,W]

    
        

#### ─────────────────────────── Add Token Mask (Same from baseline) ───────────────────────────────────── #####
       
        token_mask = torch.zeros(B_patch, N_patch, dtype=torch.bool, device=device)
        if self.hparams.mtm and self.training:
            rand = torch.rand_like(token_mask.float())
            token_mask = rand < self.hparams.p_mtm
            token_mask[:, : self.hparams.leave_n_first] = False           # keep first tokens
        token_mask = token_mask.unsqueeze(-1).expand(-1, -1, self.hparams.n_hidden)

        if self.hparams.mask_clouds:
            cloud_mask = (mask_patches.max(-1, keepdim=True)[0] > 0)      # (B_p, N_p, 1)
            cloud_mask = cloud_mask.expand(-1, -1, self.hparams.n_hidden) # (B_p, N_p, H)
        else:
            cloud_mask = torch.zeros_like(token_mask, dtype=torch.bool)

        
        mask_vec = self.mask_token.view(1, 1, -1)                         # (1,1,H) → broadcast
        full_mask = token_mask | cloud_mask                               # union of the two
        image_patches_embed = torch.where(full_mask, mask_vec, image_patches_embed)



#### ─────────────────────────── Add embedding ───────────────────────────────────── #####

        # adding embedding 
        if self.hparams.use_weather:
            patches_embed = image_patches_embed + weather_patches_embed + cap_patch  # (B_p,T,H)
        else:
            patches_embed = image_patches_embed

        # Add Positional Embedding
        pos_embed = (
            get_sinusoid_encoding_table(N_patch, self.hparams.n_hidden)
            .to(patches_embed.device)
            .unsqueeze(0)
            .repeat(B_patch, 1, 1)
        )
        x = patches_embed + pos_embed
        

#### ─────────────────────────── Transformer Block ───────────────────────────────────── #####
        # Using Contextformer 
        for blk in self.blocks:  
            x = blk(x)
       
        # Decode image patches
        x_out = self.head(x)

        

        # Mask Non-masked inputs
        x_out[
            ~token_mask.bool()[
                :,
                :,
                : self.hparams.n_out
                * self.hparams.patch_size
                * self.hparams.patch_size,
            ]
        ] = -1

        # unpatchify images
        images_out = (
            x_out.reshape(
                B,
                H // self.hparams.patch_size,
                W // self.hparams.patch_size,
                N_patch,
                self.hparams.n_out,
                self.hparams.patch_size,
                self.hparams.patch_size,
            )
            .permute(0, 3, 4, 1, 5, 2, 6)
            .reshape(B, N_patch, self.hparams.n_out, H, W)
        )
       
#### ─────────────────────────── Same from baseline code  ───────────────────────────────────── #####
        if self.hparams.add_last_ndvi:
            mask = hr_dynamic_mask[:, :c_l, ...]

            indxs = (
                torch.arange(c_l, device=mask.device)
                .expand(B, self.hparams.n_out, H, W, -1)
                .permute(0, 4, 1, 2, 3)
            )

            ndvi = hr_dynamic_inputs[:, :c_l, : self.hparams.n_out, ...]

            last_pixel = torch.gather(
                ndvi, 1, (indxs * (mask < 1)).argmax(1, keepdim=True)
            )

            images_out += last_pixel.repeat(1, N_patch, 1, 1, 1)

        elif self.hparams.add_mean_ndvi:
            mask = hr_dynamic_mask[:, :c_l, ...]
            ndvi = hr_dynamic_inputs[:, :c_l, : self.hparams.n_out, ...]

            mean_ndvi = (
                (ndvi * (mask < 1)).sum(1, keepdim=True)
                / ((mask < 1).sum(1, keepdim=True) + 1e-8)
            ).clamp(min=-1.0, max=1.0)

            images_out += mean_ndvi.repeat(1, N_patch, 1, 1, 1)

        if self.hparams.predict_delta_avg:

            image_avg = self.head_avg(x[:, :c_l, :].mean(1).unsqueeze(1))
            image_avg_out = (
                image_avg.reshape(
                    B,
                    H // self.hparams.patch_size,
                    W // self.hparams.patch_size,
                    1,
                    self.hparams.n_out,
                    self.hparams.patch_size,
                    self.hparams.patch_size,
                )
                .permute(0, 3, 4, 1, 5, 2, 6)
                .reshape(B, 1, self.hparams.n_out, H, W)
            )

            images_out += image_avg_out.repeat(1, N_patch, 1, 1, 1)

        elif self.hparams.predict_delta_max:
            image_avg = self.head_avg(x[:, :c_l, :].max(1)[0]).unsqueeze(1)
            image_avg_out = (
                image_avg.reshape(
                    B,
                    H // self.hparams.patch_size,
                    W // self.hparams.patch_size,
                    1,
                    self.hparams.n_out,
                    self.hparams.patch_size,
                    self.hparams.patch_size,
                )
                .permute(0, 3, 4, 1, 5, 2, 6)
                .reshape(B, 1, self.hparams.n_out, H, W)
            )

            images_out += image_avg_out.repeat(1, N_patch, 1, 1, 1)

        elif self.hparams.predict_delta:
            images_out[:, 0, ...] += images[:, 0, : self.hparams.n_out, ...]
            images_out = torch.cumsum(images_out, 1)
        elif self.hparams.predict_delta0:
            images_out += (images[:, :1, : self.hparams.n_out, ...]).repeat(
                1, N_patch, 1, 1, 1
            )

        if not self.training:
            images_out = images_out[:, -preds_length:]

        if self.hparams.spatial_shuffle:
            B, T, C, H, W = images_out.shape
            images_out = (
                images_out.permute(1, 2, 0, 3, 4)
                .reshape(T, C, B * H * W)[:, :, invperm]
                .reshape(T, C, B, H, W)
                .permute(2, 0, 1, 3, 4)
            )
  
        return images_out, {}
