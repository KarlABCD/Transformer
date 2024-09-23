import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from visualization import Visualization
from prepare import TranslationCorpus

d_k = 64
d_v = 64
d_embedding = 128
n_heads = 8
batch_size = 3
n_layers = 1

class ScaledDotProductionAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductionAttention,self).__init__()
    def forward(self, Q, K, V, mask):
        scores = torch.matmul(Q, K.transpose(-1,-2))/np.sqrt(d_k)
        scores.masked_fill_(mask, -1e9)
        weights = nn.Softmax(dim = -1)(scores)
        contexts = torch.matmul(weights, V)
        return contexts, weights

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.W_Q = nn.Linear(d_embedding, n_heads*d_k)
        self.W_K = nn.Linear(d_embedding, n_heads*d_k)
        self.W_V = nn.Linear(d_embedding, n_heads*d_v)
        self.Linear = nn.Linear(n_heads*d_v, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)

    def forward(self,Q,K,V,attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, weights = ScaledDotProductionAttention()( q_s, k_s, v_s, attn_mask)
        context = context.transpose(1,2).contiguous().view(batch_size, -1, n_heads*d_v)
        output = self.Linear(context)
        output = self.layer_norm(output + residual)
        return output, weights
    
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_ff=2048):
        super(PoswiseFeedForwardNet,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_embedding, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_embedding, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, inputs):
        residual = inputs
        output = nn.ReLU()(self.conv1(inputs.transpose(1,2)))
        output = self.conv2(output).transpose(1,2)
        output = self.layer_norm(output + residual)
        return output

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn_weights

class Encoder(nn.Module):
    def __init__(self, corpus):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(len(corpus.src_vocab),d_embedding)
        self.pos_emb = nn.Embedding.from_pretrained(get_sin_enc_table(corpus.src_len + 1,d_embedding),freeze = True)
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n_layers))
    def forward(self, enc_inputs):
        pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(pos_indices)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attn_weights = []
        for layer in self.layers:
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)
        return enc_outputs, enc_self_attn_weights

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer,self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Decoder(nn.Module):
    def __init__(self, corpus):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(len(corpus.tgt_vocab),d_embedding)
        self.pos_emb = nn.Embedding.from_pretrained(get_sin_enc_table(corpus.tgt_len+1,d_embedding),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
    def forward(self,dec_inputs, enc_inputs, enc_outputs):
        pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(pos_indices)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs,dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask),0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
        dec_self_attns,dec_enc_attns = [],[]
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
            #Visualization.PlotOneDimTensor(dec_outputs)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self, corpus):
        super(Transformer, self).__init__()
        self.encoder = Encoder(corpus)
        self.decoder = Decoder(corpus)
        self.projection = nn.Linear(d_embedding, len(corpus.tgt_vocab),bias = False)
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns
def forward_hook(module, input, output):
    print(f"Module name: {module.__class__.__name__}")
    forward_hook.inputs = []
    forward_hook.outputs = []
    if not isinstance(input, tuple):
        input = tuple(input)
    if not isinstance(output, tuple):
        output = tuple(output)
    for i in range(0, len(input)):
        if isinstance(input[i], torch.Tensor):
            if(input[i].device.type == 'cuda'):
                forward_hook.inputs.append(input[i].cpu().data.detach().numpy())
            else:
                forward_hook.inputs.append(input[i].data.detach().numpy())
        if isinstance(input[i], list):
            for m in range(0, len(input[i])):
                if (input[i][m].device.type == 'cuda'):
                    forward_hook.outputs.append(input[i][m].cpu().data.detach().numpy())
                else:
                    forward_hook.outputs.append(input[i][m].data.detach().numpy())
    for j in range(0, len(output)):
        if isinstance(output[j], torch.Tensor):
            if (output[j].device.type == 'cuda'):
                forward_hook.outputs.append(output[j].cpu().data.detach().numpy())
            else:
                forward_hook.outputs.append(output[j].data.detach().numpy())
        if isinstance(output[j], list):
            for m in range(0, len(output[j])):
                if(output[j][m].device.type == 'cuda'):
                    forward_hook.outputs.append(output[j][m].cpu().data.detach().numpy())
                else:
                    forward_hook.outputs.append(output[j][m].data.detach().numpy())
def get_sin_enc_table(n_position, embedding_dim):
    sinusoid_table = np.zeros((n_position,embedding_dim))
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i/np.power(10000, 2*(hid_j//2)/embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle
    sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2])
    sinusoid_table[:,1::2] = np.sin(sinusoid_table[:,1::2])
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q,seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0),seq.size(1),seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape),k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte().to(seq.device)
    return subsequent_mask
def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 5).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0,5):
        dec_input[0][i] = next_symbol
        dec_output,_,_ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_output)
        prob = projected.squeeze(0).max(dim = -1, keepdim = False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    dec_outputs = dec_input
    return dec_outputs