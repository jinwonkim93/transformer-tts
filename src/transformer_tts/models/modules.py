import torch.nn.functional as F
import torch.nn as nn
import torch
import math


    
class LinearNorm(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/layers.py"""
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/layers.py"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

    
class TextPrenet(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/model.py"""
    def __init__(self, encoder_n_convolutions=3, encoder_embedding_dim=256, encoder_kernel_size=3):
        super(TextPrenet, self).__init__()
        convolutions = []
        self.preprojection = LinearNorm(encoder_embedding_dim*2, encoder_embedding_dim*2, w_init_gain='relu')
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(encoder_embedding_dim*2,
                         encoder_embedding_dim*2,
                         kernel_size=encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(encoder_embedding_dim*2))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        self.postprojection = LinearNorm(encoder_embedding_dim*2, encoder_embedding_dim)
            
    def forward(self, x, g=None):
        x = self.preprojection(x).transpose(1,2)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = self.postprojection(x.transpose(1,2))
        return x
            

class Prenet(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/model.py"""
    def __init__(self, in_dim=80, sizes=[256,256]):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x
    
    
class Postnet(nn.Module):
    """https://github.com/NVIDIA/tacotron2/blob/master/model.py"""
    """Postnet
        - Five 1-d convolution with 1024 channels and kernel size 5
    """

    def __init__(self, n_mel_channels=80, postnet_embedding_dim=1024, postnet_kernel_size=5, postnet_n_convolutions=5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )
        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x
    
    
class PositionalEncoding(nn.Module):
    """https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.transpose(0,1))

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->ParlerTTS
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=1024, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    # Ignore copy
    @torch.no_grad()
    def forward(self, device_type, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :]
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor over which to apply the rope embeddings
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

    

class EncoderLayer(nn.TransformerEncoderLayer):
    """https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html"""
    def __init__(self, d_model, nhead=4, dim_feedforward=1024, dropout=0.1, activation="relu", 
                 layer_norm_eps=1e-05, batch_first=True, norm_first=False, device=None, dtype=None):
        super(EncoderLayer, self).__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, 
            layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first, device=device, dtype=dtype)
        
        
class DecoderLayer(nn.TransformerDecoderLayer):
    """https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoderLayer.html"""
    def __init__(self, d_model, nhead=4, dim_feedforward=1024, dropout=0.1, activation="relu", 
                 layer_norm_eps=1e-05, batch_first=True, norm_first=False, device=None, dtype=None):
        super(DecoderLayer, self).__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, 
            layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first, device=device, dtype=dtype)


class Encoder(nn.TransformerEncoder):
    """https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html"""
    def __init__(self, encoder_layer, num_layers=3, norm=None):
        super(Encoder, self).__init__(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
class Decoder(nn.TransformerDecoder):
    """https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html"""
    def __init__(self, decoder_layer, num_layers=3, norm=None):
        super(Decoder, self).__init__(decoder_layer=decoder_layer, num_layers=num_layers, norm=norm)
        self._reset_parameters()

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)