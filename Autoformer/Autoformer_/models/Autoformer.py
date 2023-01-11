import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self):
        super(Model, self).__init__()
        self.seq_len = 10
        self.label_len = 1

        self.enc_in = 5
        self. d_model = 512
        self.embed = "timeF"
        self.freq = 'h'
        self.dropout = 0.05
        self.e_layers = 3
        self.factor = 3
        self.d_ff = 512
        self.n_heads = 8

        # Decomp
        kernel_size = 3


        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in,self.d_model, self.embed, self.freq, self.dropout)


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                        output_attention=False),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    moving_avg=kernel_size,
                    dropout=self.dropout,
                    activation="relu"
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        self.linear1 = nn.Linear(512,1)
        self.linear2 = nn.Linear(10,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_enc,enc_self_mask=None):
        # decomp init
        # enc
        # print(x_enc.size()) # ([32, 10, 5])
        enc_out = self.enc_embedding(x_enc)
        # print(enc_out.size()) # ([32, 10, 512])
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        enc_out = self.linear1(enc_out).squeeze()
        enc_out = self.linear2(enc_out).squeeze()
        
        enc_out= self.sigmoid(enc_out)

        return enc_out # [B, L, D]
