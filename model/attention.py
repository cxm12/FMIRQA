import torch.nn.functional as F
import result_IQA.model.common as common
import math
import torch
from torch import nn
from einops import rearrange, repeat
from functools import partial
from contextlib import contextmanager


# helpers

def exists(val):
    return val is not None


def empty(tensor):
    return tensor.numel() == 0


def default(val, d):
    return val if exists(val) else d


@contextmanager
def null_context():
    yield


def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val


def get_module_device(module):
    return next(module.parameters()).device


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Always(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val
    
    def forward(self, *args, **kwargs):
        return self.val


# kernel functions

# transcribed from jax to pytorch from

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=False, eps=1e-4, device=None):
    b, h, *_ = data.shape
    
    # data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
    
    ratio = (projection_matrix.shape[0] ** -0.5)
    
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)
    
    # data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    data_dash = torch.einsum('...id,...jd->...ij', data, projection)
    
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0)
    diag_data = diag_data.unsqueeze(dim=-1)
    
    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data) + eps)
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data) + eps)
    
    return data_dash.type_as(data)


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True,
                       device=None):
    b, h, *_ = data.shape
    
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.
    
    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon
    
    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)
    projection = projection.type_as(data)
    
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)
    
    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime.type_as(data)


def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)
    
    block_list = []
    
    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)
    
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q[:remaining_rows])
    
    final_matrix = torch.cat(block_list)
    
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')
    
    return torch.diag(multiplier) @ final_matrix


# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


class ENLA(nn.Module):
    def __init__(self, dim_heads, nb_features=None, ortho_scaling=0, generalized_attention=False, kernel_fn=nn.ReLU(),
                 no_projection=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))
        
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=self.nb_features,
                                         nb_columns=dim_heads, scaling=ortho_scaling)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)
        
        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn
        
        # if this is turned on, no projection will be used
        # queries and keys will be softmax-ed as in the original efficient attention paper
        self.no_projection = no_projection
    
    @torch.no_grad()
    def redraw_projection_matrix(self, device):
        projections = self.create_projection(device=device)
        self.projection_matrix.copy_(projections)
        del projections
    
    def forward(self, q, k, v):
        # q[b,h,n,d],b is batch ,h is multi head, n is number of batch, d is feature
        device = q.device
        
        if self.no_projection:
            q = q.softmax(dim=-1)
            k = k.softmax(dim=-2)
        
        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn=self.kernel_fn,
                                    projection_matrix=self.projection_matrix, device=device)
            q, k = map(create_kernel, (q, k))
        
        else:
            create_kernel = partial(softmax_kernel, projection_matrix=self.projection_matrix, device=device)
            q = create_kernel(q, is_query=True)
            k = create_kernel(k, is_query=False)
        
        attn_fn = linear_attention
        out = attn_fn(q, k, v)
        return out


# a module for keeping track of when to update the projections

class ProjectionUpdater(nn.Module):
    def __init__(self, instance, feature_redraw_interval):
        super().__init__()
        self.instance = instance
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))
    
    def fix_projections_(self):
        self.feature_redraw_interval = None
    
    def redraw_projections(self):
        model = self.instance
        
        if not self.training:
            return
        
        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(model)
            
            fast_attentions = find_modules(model, ENLA)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)
            
            self.calls_since_last_redraw.zero_()
            return
        
        self.calls_since_last_redraw += 1
    
    def forward(self, x):
        raise NotImplemented


class ENLCA(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv, res_scale=0.1):
        super(ENLCA, self).__init__()
        self.conv_match1 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_match2 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=None)
        self.res_scale = res_scale
        self.attn_fn = ENLA(
            dim_heads=channel // reduction,
            nb_features=128,
        )
        self.k = math.sqrt(6)

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)  #[B,C,H,W]
        x_embed_2 = F.normalize(x_embed_2, p=2, dim=1,eps=5e-5)*self.k
        x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5)*self.k
        N, C, H, W = x_embed_1.shape
        loss = 0
        if self.training:
            score = torch.matmul(x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C)),
                                 x_embed_2.view(N, C, H * W))  # [N,H*W,H*W]
            score = torch.exp(score)
            score = torch.sort(score, dim=2, descending=True)[0]
            positive = torch.mean(score[:, :, :15], dim=2)
            negative = torch.mean(score[:, :, 50:65], dim=2)  # [N,H*W]
            loss = F.relu(-1 * torch.log(positive / (negative + 1e-6))+1)
            loss = torch.mean(loss)

        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view(N, 1, H * W, C)
        x_embed_2 = x_embed_2.permute(0, 2, 3, 1).view(N, 1, H * W, C)
        x_assembly = x_assembly.permute(0, 2, 3, 1).view(N, 1, H*W, -1 )
        x_final = self.attn_fn(x_embed_1, x_embed_2, x_assembly).squeeze(1)  # (1, H*W, C)
        return x_final.permute(0, 2, 1).view(N, -1, H, W)*self.res_scale+input, loss


class ENLCA3D(nn.Module):
    def __init__(self, channel=128, reduction=2, ksize=3, scale=3, stride=1, softmax_scale=10, average=True,
                 conv=common.default_conv3D, res_scale=0.1):
        super(ENLCA3D, self).__init__()
        self.conv_match1 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_match2 = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=None)
        self.res_scale = res_scale
        self.attn_fn = ENLA(
            dim_heads=channel // reduction,
            nb_features=128,
        )
        self.k = math.sqrt(6)

    def forward(self, input):
        x_embed_1 = self.conv_match1(input)
        x_embed_2 = self.conv_match2(input)
        x_assembly = self.conv_assembly(input)  # [B,C,H,W]
        x_embed_2 = F.normalize(x_embed_2, p=2, dim=1, eps=5e-5)*self.k
        x_embed_1 = F.normalize(x_embed_1, p=2, dim=1, eps=5e-5)*self.k
        N, C1, Z, H, W = x_assembly.shape
        x_assembly = x_assembly.view(N, C1*Z, H, W)
        N, C0, Z, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.view(N, C0*Z, H, W)
        x_embed_2 = x_embed_2.view(N, C0*Z, H, W)
        x_embed_2 = x_embed_2.view(N, C0*Z, H, W)
        N, C, H, W = x_embed_1.shape
        loss = 0
        if self.training:
            score = torch.matmul(x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C)),
                                 x_embed_2.view(N, C, H * W))  # [N,H*W,H*W]
            score = torch.exp(score)
            score = torch.sort(score, dim=2, descending=True)[0]
            positive = torch.mean(score[:, :, :15], dim=2)
            negative = torch.mean(score[:, :, 50:65], dim=2)  # [N,H*W]
            loss = F.relu(-1 * torch.log(positive / (negative + 1e-6))+1)
            loss = torch.mean(loss)

        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view(N, 1, H * W, C)
        x_embed_2 = x_embed_2.permute(0, 2, 3, 1).view(N, 1, H * W, C)
        x_assembly = x_assembly.permute(0, 2, 3, 1).view(N, 1, H*W, -1)
        x_final = self.attn_fn(x_embed_1, x_embed_2, x_assembly).squeeze(1)  # (1, H*W, C)
        
        out = x_final.permute(0, 2, 1).view(N, -1, H, W)*self.res_scale

        out = out.view(N, C1, Z, H, W) + input
        return out, loss
