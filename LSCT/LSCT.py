import torch
import torch.nn as nn
import torch.nn.functional as F
from VIT import Transformer




class VideoQualityTransformer(nn.Module):
    def __init__(self,num_layers,
        d_model,
        num_heads,
        mlp_dim,
        dropout=0.1,
        maximum_position_encoding=6000):
        super(VideoQualityTransformer, self).__init__()
        self.feature_proj=nn.Linear(64,64)
        self.pos_emb=nn.Parameter(torch.randn(1,maximum_position_encoding,d_model))
        self.quality_emb=nn.Parameter(torch.randn(1,1,d_model))
        self.dropout=nn.Dropout(dropout)
        self.enc_layers=Transformer(d_model,num_layers,num_heads,d_model,mlp_dim,dropout)
        self.mlp_head=nn.Sequential(
            nn.Linear(d_model,mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,1)
        )



    def forward(self,x):
        batch_size=x.shape[0]
        # mask=create_padding_mask(x)

        clip_length=x.shape[1]
        x=self.feature_proj(x)
        # print(torch.mean(x.clone().detach()).item())
        quality_emb=self.quality_emb.expand(batch_size, -1 ,-1)
        x=torch.cat([quality_emb,x],dim=-2)
        # print(torch.mean(x.clone().detach()).item())
        x=x+self.pos_emb[:,:clip_length+1]
        # print(torch.mean(x.clone().detach()).item())
        x=self.dropout(x)

        x=self.enc_layers(x)
        # print(torch.mean(x.clone().detach()).item())
        x = self.mlp_head(x[:, 0])
        # print(torch.mean(x.clone().detach()).item())
        return x



class LSCT(nn.Module):
    def __init__(self,clip_length=16, feature_length=1280, cnn_filters=(32, 64), pooling_sizes=(4, 4),
                 transformer_params=(2, 64, 4, 64), strides=1, dropout_rate=0.1):
        super(LSCT, self).__init__()
        self.cnn1d=nn.Sequential(nn.Conv1d(feature_length,cnn_filters[0],kernel_size=3,stride=strides,padding=1),nn.MaxPool1d(kernel_size=pooling_sizes[0]),nn.Dropout(dropout_rate),
                                 nn.Conv1d(cnn_filters[0],cnn_filters[1],kernel_size=3,stride=strides,padding=1),nn.MaxPool1d(kernel_size=pooling_sizes[1]),nn.Dropout(dropout_rate))
        self.transformer= VideoQualityTransformer(
        num_layers=transformer_params[0],
        d_model=transformer_params[1],
        num_heads=transformer_params[2],
        mlp_dim=transformer_params[3],
        dropout=dropout_rate,
    )

    def forward(self,x):
        batch_size=x.shape[0]
        num_clips=x.shape[1]
        x=x.view(batch_size*num_clips,16,1280).transpose(1,2)
        
        x=self.cnn1d(x).squeeze(-1)
        # print(torch.mean(x.clone().detach()).item())
        x=x.view(batch_size,num_clips,64)
        x=self.transformer(x)
        return x.view(-1)

