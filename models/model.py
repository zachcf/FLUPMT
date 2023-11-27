import torch
import torch.nn as nn
from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from .decoder import Decoder, DecoderLayer
from .attn import FullAttention, ProbAttention, AttentionLayer
from .embed import DataEmbedding
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
class Vocab:

    def __init__(self):
        self.t = ['A', 'F', 'Q', 'R', 'T', 'Y', 'V', 'I', 'H', 'K', 'P', 'N', 'E', 'G', 'S', 'M', 'D', 'W', 'C',
                           'L', '-', 'B', 'J', 'Z', 'X']
        self.token2idx = {v:i  for i,v in enumerate(self.t)}
        self.idx2token = {i:v  for i, v in enumerate(self.t)}
    def __getitem__(self, tokens_or_indices):

        if isinstance(tokens_or_indices, (str, int)):
            return self.token2idx.get(tokens_or_indices, 0) if isinstance(
                tokens_or_indices, str) else self.idx2token.get(tokens_or_indices, 'X')

        elif isinstance(tokens_or_indices, (list, tuple)):
            return [self.__getitem__(item) for item in tokens_or_indices]
        else:
            raise TypeError

    def __len__(self):
        return len(self.idx2token)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_feats)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.float()
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GCNDecoder(torch.nn.Module):
    def __init__(self,out_feats):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * out_feats, out_feats),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_feats, 1)
        )
    def forward(self, x,edge_index):
        x_src = x[edge_index[0]]
        x_dst = x[edge_index[1]]
        edge_x = torch.cat((x_src, x_dst), dim=1)
        out = self.fc(edge_x)
        out = torch.flatten(out)
        return out
class LSTFEncoder(nn.Module):
    def __init__(self, enc_in,factor=5, d_model=512, n_heads=8, e_layers=3,  d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu',
                 output_attention=False,
                 device=torch.device('cuda:0')):
        super().__init__()
        self.attn = attn
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        Attn = ProbAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x_enc,  enc_self_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        return enc_out
class LSTFDecoder(nn.Module):
    def __init__(self, dec_in, c_out,out_len,
                 factor=5, d_model=512, n_heads=8,  d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu',
                 device=torch.device('cuda:0')):
        super().__init__()
        self.pred_len = out_len
        self.attn = attn
        Attn = ProbAttention
        self.dec_embedding = DataEmbedding(dec_in, d_model, dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, enc_out, x_dec,  dec_self_mask=None, dec_enc_mask=None):
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

class Expert_net(nn.Module):
    def __init__(self, feature_dim, expert_dim):
        super(Expert_net, self).__init__()

        p = 0.1
        self.dnn_layer = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, expert_dim),
            nn.ReLU(),
            nn.Dropout(p)
        )

    def forward(self, x):
        out = self.dnn_layer(x)
        return out

class Extraction_Network(nn.Module):


    def __init__(self, FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum):
        super(Extraction_Network, self).__init__()

        self.GateNum = GateNum  

        self.n_task = 2
        self.n_share = 1

        self.Exper_Layer = Expert_net(FeatureDim, ExpertOutDim)
        self.seqExpert = LSTFEncoder(enc_in=56400)
        self.CNNExpert = GCNEncoder(56400, 200, 512)

        self.Experts_A = [self.seqExpert for i in range(TaskExpertNum)] 

        # self.seqExpert = self.seqLayer

        self.Experts_Shared = [self.Exper_Layer for i in
                               range(CommonExpertNum)]  

        self.Experts_B = [self.CNNExpert for i in range(TaskExpertNum)]  


        self.Task_Gate_Layer = nn.Sequential(nn.Linear(FeatureDim, TaskExpertNum + CommonExpertNum),
                                             nn.Softmax(dim=1))
        self.Task_Gates = [self.Task_Gate_Layer for i in range(self.n_task)]  

        self.Shared_Gate_Layer = nn.Sequential(nn.Linear(FeatureDim, 2 * TaskExpertNum + CommonExpertNum),
                                               nn.Softmax(dim=1))
        self.Shared_Gates = [self.Shared_Gate_Layer for i in range(self.n_share)] 

    def forward(self, x_A, x_B):
        bat = x_A.shape[0]
        Experts_A_Out = [expert(x_A) for expert in self.Experts_A]  
        Experts_A_Out= torch.cat(([expert[:, :,np.newaxis, :] for expert in Experts_A_Out]),dim=1)


        x_A = x_A.reshape(-1,x_A.shape[-1]).float()

        x_S = torch.cat([x_A,x_B.x],dim=0).float()
        Experts_A_Out = Experts_A_Out.reshape(-1,Experts_A_Out.shape[-2],Experts_A_Out.shape[-1])
        la = Experts_A_Out.shape[0]
        lb = x_A.shape[0]
        Experts_Shared_Out = [expert(x_S) for expert in self.Experts_Shared]  
        Experts_Shared_Out = torch.cat(([expert[:, np.newaxis, :] for expert in Experts_Shared_Out]),
                                       dim=1)  # (bs,CommonExpertNum,ExpertOutDim)(175,1,512)

        Experts_B_Out = [expert(x_B) for expert in self.Experts_B]  
        Experts_B_Out = torch.cat(([expert[:, np.newaxis, :] for expert in Experts_B_Out]),
                                  dim=1)
        #  (bs,TaskExpertNum(1),ExpertOutDim)(170,1,512)
        Gate_A = self.Task_Gates[0](x_A)  # (bs,TaskExpertNum+CommonExpertNum)5,



        Gate_B = self.Task_Gates[1](x_B.x)  #  (bs,TaskExpertNum+CommonExpertNum)170,

        '''GateA out'''
        #l,bs,TaskExpertNum,ExpertOutDim
        g = Gate_A.unsqueeze(2) # (bs,TaskExpertNum+CommonExpertNum,1)
        experts = torch.cat([Experts_A_Out, Experts_Shared_Out[:la]],
                            dim=1)  # (bs,TaskExpertNum+CommonExpertNum,ExpertOutDim)

        Gate_A_Out = torch.matmul(experts.transpose(1,2),g[:la])  # (bs,ExpertOutDim,1)
        Gate_A_Out = Gate_A_Out.squeeze(2)  # (bs,ExpertOutDim)
        Gate_A_Out = Gate_A_Out.reshape(bat,-1,Gate_A_Out.shape[-1])

        '''GateB out'''
        g = Gate_B.unsqueeze(2)  # (bs,TaskExpertNum+CommonExpertNum,1)
        experts = torch.cat([Experts_B_Out, Experts_Shared_Out[lb:]],
                            dim=1)  # (bs,TaskExpertNum+CommonExpertNum,ExpertOutDim)
        Gate_B_Out = torch.matmul(experts.transpose(1, 2), g)  # (bs,ExpertOutDim,1)
        Gate_B_Out = Gate_B_Out.squeeze(2)  # (bs,ExpertOutDim)


        return Gate_A_Out, (Gate_B_Out,x_B.edge_index)


class FluPMT(nn.Module):

    def __init__(self, FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, out_len,seq_len,label_len,n_task=2):
        super(FluPMT, self).__init__()


        # self.Extraction_layer1 = Extraction_Network(FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum=3)
        self.CGC = Extraction_Network(FeatureDim, ExpertOutDim, TaskExpertNum, CommonExpertNum, GateNum=2)

        '''TowerA'''
        p1 = 0
        # hidden_layer1 = [64, 32]
        self.tower1 = LSTFDecoder(dec_in=566,c_out=566,out_len=out_len)
        '''TowerB'''
        p2 = 0
        # hidden_layer2 = [64, 32]
        self.tower2 = GCNDecoder(512)

    def forward(self, encoder_inputs, decoder_inputs,CNN_input):
        # Output_A, Output_Shared, Output_B = self.Extraction_layer1(encoder_inputs,CNN_input)
        # Gate_A_Out, Gate_B_Out = self.CGC(Output_A,Output_B,Output_Shared )
        Gate_A_Out, Gate_B_Out = self.CGC(encoder_inputs,CNN_input)
        out1 = self.tower1(Gate_A_Out,decoder_inputs)
        out2 = self.tower2(*Gate_B_Out)

        return out1, out2


class UncertaintyLoss(nn.Module):

    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        self.sigma = nn.Parameter(torch.ones(v_num))
        self.v_num = v_num

    def forward(self, *input):
        loss = 0
        for i in range(self.v_num):
            loss += input[i] / (2 * self.sigma[i] ** 2)
        # sigma_clamped = torch.clamp(self.sigma, min=0.001)
        loss += torch.log(self.sigma.prod())
        return loss
