import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

class Model(nn.Module):
    """
    Transformer-based Time Series Forecasting Model.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # 120 (Input sequence length)
        self.pred_len = configs.pred_len  # 30 (Prediction length)
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding Layer
        self.enc_embedding = DataEmbedding_inverted(
            c_in=configs.c_in,   # 7 weather features
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
    
    

