import sys
sys.path.append('/lustre/home/hyin/workspace/AI4S/paddle_project/utils')
import paddle_aux
import paddle
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from layers.GAT import GAT_Encoder
from layers.SelfAttention_Family import ProbAttention, AttentionLayer, FullAttention
import argparse


def FFT_for_Period(x, k=2):
    xf = paddle.fft.rfft(x=x, axis=1)
    frequency_list = abs(xf).mean(axis=0).mean(axis=-1)
    frequency_list[0] = 0
    _, top_list = paddle.topk(k=k, x=frequency_list)
    top_list = top_list.detach().cpu().numpy()
    period = tuple(x.shape)[1] // top_list
    return period, abs(xf).mean(axis=-1)[:, top_list]


class TimesBlock(paddle.nn.Layer):

    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = paddle.nn.Sequential(Inception_Block_V1(configs.d_model,
            configs.d_ff, num_kernels=configs.num_kernels), paddle.nn.GELU(
            ), Inception_Block_V1(configs.d_ff, configs.d_model,
            num_kernels=configs.num_kernels))

    def forward(self, x):
        B, T, N = tuple(x.shape)
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = ((self.seq_len + self.pred_len) // period + 1
                    ) * period
                padding = paddle.zeros(shape=[tuple(x.shape)[0], length - (
                    self.seq_len + self.pred_len), tuple(x.shape)[2]]).to(x
                    .place)
                out = paddle.concat(x=[x, padding], axis=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = out.reshape(B, length // period, period, N).transpose(perm
                =[0, 3, 1, 2])
            out = self.conv(out)
            out = out.transpose(perm=[0, 2, 3, 1]).reshape(B, -1, N)
            res.append(out[:, :self.seq_len + self.pred_len, :])
        res = paddle.stack(x=res, axis=-1)
        period_weight_raw = period_weight
        period_weight = paddle.nn.functional.softmax(x=period_weight, axis=1)
        period_weight = period_weight.unsqueeze(axis=1).unsqueeze(axis=1
            ).repeat(1, T, N, 1)
        res = paddle.sum(x=res * period_weight, axis=-1)
        res = res + x
        return res, period_list, period_weight_raw


class Gat_TimesNet_mm(paddle.nn.Layer):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs, **kwargs):
        super(Gat_TimesNet_mm, self).__init__()
        configs = argparse.Namespace(**configs)
        self.device = str('cuda').replace('cuda', 'gpu')
        self.configs = configs
        self.task_name = configs.task_name
        if hasattr(configs, 'output_attention'):
            self.output_attention = configs.output_attention
        else:
            self.output_attention = False
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.dec_in = configs.dec_in
        self.gat_embed_dim = configs.enc_in
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model,
            configs.embed, configs.freq, configs.dropout)
        self.aq_gat_node_num = configs.aq_gat_node_num
        self.aq_gat_node_features = configs.aq_gat_node_features
        self.aq_GAT = GAT_Encoder(configs.aq_gat_node_features, configs.
            gat_hidden_dim, configs.gat_edge_dim, self.gat_embed_dim,
            configs.dropout).to(self.device)
        self.mete_gat_node_num = configs.mete_gat_node_num
        self.mete_gat_node_features = configs.mete_gat_node_features
        self.mete_GAT = GAT_Encoder(configs.mete_gat_node_features, configs
            .gat_hidden_dim, configs.gat_edge_dim, self.gat_embed_dim,
            configs.dropout).to(self.device)
        self.pos_fc = paddle.nn.Linear(in_features=2, out_features=configs.
            gat_embed_dim, bias_attr=True)
        self.fusion_Attention = AttentionLayer(FullAttention(False, configs
            .factor, attention_dropout=configs.dropout, output_attention=
            self.output_attention), configs.gat_embed_dim, configs.n_heads)
        self.model = paddle.nn.LayerList(sublayers=[TimesBlock(configs) for
            _ in range(configs.e_layers)])
        self.layer = configs.e_layers
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=configs.d_model)
        self.predict_linear = paddle.nn.Linear(in_features=self.seq_len,
            out_features=self.pred_len + self.seq_len)
        self.projection = paddle.nn.Linear(in_features=configs.d_model,
            out_features=configs.c_out, bias_attr=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(axis=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = paddle.sqrt(x=paddle.var(x=x_enc, axis=1, keepdim=True,
            unbiased=False) + 1e-05)
        x_enc /= stdev
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.transpose(perm=[0, 2, 1])
            ).transpose(perm=[0, 2, 1])
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(axis=1).repeat(1, self
            .pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(axis=1).repeat(1, self
            .pred_len + self.seq_len, 1)
        return dec_out

    def aq_gat(self, G):
        x = G.x[:, -self.aq_gat_node_features:].to(self.device)
        edge_index = G.edge_index.to(self.device)
        edge_attr = G.edge_attr.to(self.device)
        g_batch = G.batch.to(self.device)
        batch_size = int(len(g_batch) / self.aq_gat_node_num / self.seq_len)
        gat_output = self.aq_GAT(x, edge_index, edge_attr, g_batch)
        gat_output = gat_output.reshape((batch_size, self.seq_len, self.
            aq_gat_node_num, self.gat_embed_dim))
        gat_output = paddle.flatten(x=gat_output, start_axis=0, stop_axis=1)
        return gat_output

    def mete_gat(self, G):
        x = G.x[:, -self.mete_gat_node_features:].to(self.device)
        edge_index = G.edge_index.to(self.device)
        edge_attr = G.edge_attr.to(self.device)
        g_batch = G.batch.to(self.device)
        batch_size = int(len(g_batch) / self.mete_gat_node_num / self.seq_len)
        gat_output = self.mete_GAT(x, edge_index, edge_attr, g_batch)
        gat_output = gat_output.reshape((batch_size, self.seq_len, self.
            mete_gat_node_num, self.gat_embed_dim))
        gat_output = paddle.flatten(x=gat_output, start_axis=0, stop_axis=1)
        return gat_output

    def norm_pos(self, A, B):
>>>>>>        A_mean = torch.mean(A, axis=0)
>>>>>>        A_std = torch.std(A, axis=0)
        A_norm = (A - A_mean) / A_std
        B_norm = (B - A_mean) / A_std
        return A_norm, B_norm

    def forward(self, Data, mask=None):
        aq_G = Data['aq_G']
        mete_G = Data['mete_G']
        aq_gat_output = self.aq_gat(aq_G)
        mete_gat_output = self.mete_gat(mete_G)
        aq_pos, mete_pos = self.norm_pos(aq_G.pos.to(self.device), mete_G.
            pos.to(self.device))
        aq_pos = self.pos_fc(aq_pos).view(-1, self.aq_gat_node_num, self.
            gat_embed_dim)
        mete_pos = self.pos_fc(mete_pos).view(-1, self.mete_gat_node_num,
            self.gat_embed_dim)
        fusion_out, attn = self.fusion_Attention(aq_pos, mete_pos,
            mete_gat_output, attn_mask=None)
        aq_gat_output = aq_gat_output + fusion_out
        aq_gat_output = aq_gat_output.view(-1, self.seq_len, self.
            aq_gat_node_num, self.gat_embed_dim)
        x = aq_gat_output
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        aq_gat_output = paddle.transpose(x=x, perm=perm_0)
        aq_gat_output = paddle.flatten(x=aq_gat_output, start_axis=0,
            stop_axis=1)
        train_data = Data['aq_train_data']
        x = train_data
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        train_data = paddle.transpose(x=x, perm=perm_1)
        train_data = paddle.flatten(x=train_data, start_axis=0, stop_axis=1)
        x_enc = train_data[:, :self.seq_len, -self.dec_in:]
        x_mark_enc = train_data[:, :self.seq_len, 1:6]
        x_dec = paddle.zeros_like(x=train_data[:, -self.pred_len:, -self.
            dec_in:]).astype(dtype='float32')
        x_dec = paddle.concat(x=[train_data[:, self.seq_len - self.
            label_len:self.seq_len, -self.dec_in:], x_dec], axis=1).astype(
            dtype='float32').to(self.device)
        x_mark_dec = train_data[:, -self.pred_len - self.label_len:, 1:6]
        means = x_enc.mean(axis=1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = paddle.sqrt(x=paddle.var(x=x_enc, axis=1, keepdim=True,
            unbiased=False) + 1e-05)
        x_enc /= stdev
        enc_out = self.enc_embedding(aq_gat_output, x_mark_enc)
        enc_out = self.predict_linear(enc_out.transpose(perm=[0, 2, 1])
            ).transpose(perm=[0, 2, 1])
        for i in range(self.layer):
            enc_out, period_list, period_weight = self.model[i](enc_out)
            enc_out = self.layer_norm(enc_out)
        dec_out = self.projection(enc_out)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(axis=1).repeat(1, self
            .pred_len + self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(axis=1).repeat(1, self
            .pred_len + self.seq_len, 1)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attn
        else:
            return dec_out[:, -self.pred_len:, :], period_list, period_weight
