import torch
import math
from torch import dtype, nn
from typing import Callable


class ViTSelfAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attention_dropout: float,
                 dropout: float,
                 bias: bool = True,
                 dtype: dtype = None,
                 init_method: str = 'torch'):
        super().__init__()

        self.query = nn.Linear(dim, dim*num_heads, bias=bias)
        #self.value = nn.Linear(dim, dim*num_heads, bias=bias)

        self.attention_dropout = nn.Dropout(attention_dropout)
        self.dense = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dim = dim
        self.attention_head = num_heads
        self.gumbel_softmax = gumbel_softmax



    def forward(self, x1, c):
        q = self.query(x1)
        C = torch.cat(([c] * self.attention_head), 1)
        x = torch.matmul(q, C.transpose(0,1))
        x = x / math.sqrt(self.attention_dim)
        x = self.softmax(x)
        x = self.gumbel_softmax(x,0.1,True)
        x = self.attention_dropout(x)
        #v = self.value(c)
        x = torch.matmul(x.transpose(0,1), x1)
        x = self.dense(x)
        x = self.dropout(x)
        return x1,x



class ViTMLP(nn.Module):
    def __init__(self,
                 dim: int,
                 mlp_ratio: int,
                 activation: Callable,
                 dropout: float,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        self.dense_1 = nn.Linear(dim,
                                     mlp_ratio * dim,
                                     bias=bias)
        self.activation = activation
        self.dropout_1 = nn.Dropout(dropout)
        self.dense_2 = nn.Linear(mlp_ratio * dim,
                                     dim,
                                     bias=bias)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        return x

class ViTHead(nn.Module):
    def __init__(self,
                 dim: int,
                 num_classes: int,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 init_method: str = 'torch'):
        super().__init__()
        if representation_size:
            self.representation = nn.Linear(dim,
                                                representation_size,
                                                bias=bias)
        else:
            self.representation = None
            representation_size = dim

        self.dense = nn.Linear(representation_size,
                                       num_classes,
                                       bias=bias)

    def forward(self, x):
        if self.representation is not None:
            x = self.representation(x)
        x = self.dense(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int,
                 activation: Callable,
                 attention_dropout: float = 0.,
                 dropout: float = 0.,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon)#.cuda()
        self.attn = ViTSelfAttention(dim=dim,
                                     num_heads=num_heads,
                                     attention_dropout=attention_dropout,
                                     dropout=dropout,
                                     bias=bias,
                                     init_method=init_method)#.cuda()
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon)#.cuda()
        self.mlp = ViTMLP(dim=dim,
                          mlp_ratio=mlp_ratio,
                          activation=activation,
                          dropout=dropout,
                          bias=bias,
                          init_method=init_method)#.cuda()

    def forward(self, x, c):
        x1,c1 = self.attn(self.norm1(x), c)
        x = x + self.drop_path(x1)
        c = c + self.drop_path(c1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = x + self.drop_path(self.attn(self.norm1(x),c))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x,c


class VisionTransformer(nn.Module):
    def __init__(self,
                 dim: int = 768,
                 num_classes: int = 1000,
                 depth: int = 12,
                 num_heads: int = 4,
                 mlp_ratio: int = 4,
                 attention_dropout: float = 0.,
                 dropout: float = 0.1,
                 drop_path: float = 0.,
                 layernorm_epsilon: float = 1e-6,
                 activation: Callable = nn.functional.gelu,
                 representation_size: int = None,
                 dtype: dtype = None,
                 bias: bool = True,
                 checkpoint: bool = False,
                 init_method: str = 'torch'):
        super().__init__()


        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = [
            ViTBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=dpr[i],
                activation=activation,
                dtype=dtype,
                bias=bias,
                checkpoint=checkpoint,
                init_method=init_method,
            ) for i in range(depth)
        ]

        self.norm = nn.LayerNorm(normalized_shape=dim, eps=layernorm_epsilon)

        self.head = ViTHead(dim=dim,
                       num_classes=num_classes,
                       representation_size=representation_size,
                       dtype=dtype,
                       bias=bias,
                       init_method=init_method)



    def forward(self, x, c):
        for attn in self.blocks:
            x,c = attn(x,c)
        x = self.norm(x)
        x = self.head(x)
        return x,c


def gumbel_softmax(logits, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def k_max_vit(dim=768, num_classes=10, depth=1, num_heads=1, mlp_ratio=2, **kwargs):
    return VisionTransformer(dim, num_classes, depth, num_heads, mlp_ratio, **kwargs)