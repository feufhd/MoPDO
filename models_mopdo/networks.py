import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch.nn.functional as F
try:
    from . import configs
except ImportError:  # pragma: no cover
    from models_mopdo import configs

import os
from torchvision.utils import save_image


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class HCO(nn.Module):
    """
    du/dt -k(d2u/dx2 + d2u/dy2) = 0;
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}
    core = cos(n\pi/ax)cos(m\pi/by)exp(-[(n\pi/a)^2 + (m\pi/b)^2]kt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }
    
    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    => 
    (\phi(x, y) = linear(dwconv(input(x, y)))),
    A(n, m) = DCT2D(\phi(x, y)),
    u(x, y, t) = IDCT2D(A(n, m) * exp(-[(n\pi/a)^2 + (m\pi/b)^2])**kt)
    """
    
    branch_names = ("heat", "wave", "poisson")

    def __init__(self, res=14, dim=768, num_heads=12, has_cls_token=True, heat_k_init_value=None, wave_k_init_value=None, possion_k_init_value=None, **kwargs):
        super().__init__()
        self.res = res
        self.dim = dim
        self.num_heads = num_heads
        self.has_cls_token = has_cls_token
        
        shape = (self.res, self.res, self.num_heads)

        def as_parameter(value):
            if value is None:
                value = torch.ones(shape)
            if tuple(value.shape) != shape:
                raise ValueError(f"shared PDE coefficient has shape {value.shape}, expected {shape}")
            if isinstance(value, nn.Parameter):
                return value
            return nn.Parameter(value, requires_grad=True)

        # The encoder passes the same Parameter objects to every block. Assigning
        # them here registers the shared tensors with PyTorch for optimization,
        # device moves, and checkpoints.
        self.heat_k = as_parameter(heat_k_init_value)
        self.wave_k = as_parameter(wave_k_init_value)
        self.possion_k = as_parameter(possion_k_init_value)

        # ����ʱ��Ͳ�����ʼ��
        self.heat_time = nn.Parameter(torch.zeros(1, 1, dim // num_heads), requires_grad=True)
        self.wave_time = nn.Parameter(torch.zeros(1, 1, dim // num_heads), requires_grad=True)
        self.possion_time = nn.Parameter(torch.zeros(1, 1, dim // num_heads), requires_grad=True)
        
        self.path_weight = nn.Parameter(torch.randn(3, num_heads), requires_grad=True)
        self.tem = nn.Parameter(5.0*torch.ones(num_heads), requires_grad=True)

        # ������ʼ��
        nn.init.normal_(self.heat_time, mean=0.05, std=.01)
        nn.init.normal_(self.wave_time, mean=0.05, std=.01)
        nn.init.normal_(self.possion_time, mean=0.5, std=.1)
        
        #nn.init.normal_(self.tem, mean=2.0, std=0.5)

        self.relu = nn.ReLU()
        self.active_branches = (0, 1, 2)

    def set_active_branches(self, branches):
        """Select PDE responses used by this layer (default: all three)."""
        if isinstance(branches, str):
            branches = [item.strip() for item in branches.split(",") if item.strip()]
        aliases = {"possion": "poisson"}
        indices = []
        for branch in branches:
            if isinstance(branch, str):
                branch = aliases.get(branch.lower(), branch.lower())
                if branch not in self.branch_names:
                    raise ValueError(
                        f"unknown HCO branch {branch!r}; expected {self.branch_names}"
                    )
                index = self.branch_names.index(branch)
            else:
                index = int(branch)
                if index < 0 or index >= len(self.branch_names):
                    raise ValueError(f"HCO branch index out of range: {index}")
            if index not in indices:
                indices.append(index)
        if not indices:
            raise ValueError("at least one HCO branch must be active")
        self.active_branches = tuple(sorted(indices))
        return self


    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # exp(-[(n\pi/a)^2 + (m\pi/b)^2])
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        #weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, is_training=False, epoch=0, max_epoch=110, entropy_scale=0.1):
        B, L, C = x.shape
        
        H, W = int(L**0.5), int(L**0.5)

        #nh = self.num_heads
        c_nh = int(self.dim // self.num_heads)

        if self.has_cls_token:
            cls_token = x[:, 0]
            x = x[:, 1:]
            
        x = x.contiguous().view(B, H, W, C)
        
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
        
        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)
        
        branch_logits = self.path_weight / self.tem
        if len(self.active_branches) != len(self.branch_names):
            active_mask = torch.zeros(
                len(self.branch_names), dtype=torch.bool, device=branch_logits.device
            )
            active_mask[list(self.active_branches)] = True
            branch_logits = branch_logits.masked_fill(
                ~active_mask[:, None], torch.finfo(branch_logits.dtype).min
            )
        weight = torch.softmax(branch_logits, 0)
        entropy = None
        if is_training:
            active_weight = weight[list(self.active_branches)]
            entropy = -torch.mean(active_weight * torch.log(active_weight + 1e-8))
            decay_factor = (max(1 - (epoch / max_epoch), 0.5) - 0.5) / 0.5
            lambda_entropy = entropy_scale * decay_factor
            # Added to the training loss, so the negative sign rewards high path
            # entropy early and decays the exploration pressure to zero halfway.
            entropy = -lambda_entropy * entropy
        
        heat_k = self.heat_k
        wave_k = self.wave_k
        possion_k = self.possion_k
        
        #import pdb
        #pdb.set_trace()
        
        response = torch.zeros((N, M, C), dtype=x.dtype, device=x.device)
        if 0 in self.active_branches:
            response = response + weight[0].repeat_interleave(c_nh) * torch.pow(
                torch.exp(-weight_exp[:, :, None]),
                self.relu((heat_k.unsqueeze(-1) * self.heat_time).reshape(N, M, C)),
            )
        if 1 in self.active_branches:
            response = response + weight[1].repeat_interleave(c_nh) * torch.cos(
                weight_exp[:, :, None]
                * self.relu((wave_k.unsqueeze(-1) * self.wave_time).reshape(N, M, C))
            )
        if 2 in self.active_branches:
            response = response + weight[2].repeat_interleave(c_nh) * self.relu(
                (possion_k.unsqueeze(-1) * self.possion_time).reshape(N, M, C)
            )

        x = torch.einsum("bnmc,nmc -> bnmc", x, response)
        
        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)

        x = x.contiguous().view(B, H*W, C)
        
        if self.has_cls_token:
            x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        
        if is_training:
            return x, entropy
        return x


class MoPDOLinear(nn.Module):
    def __init__(self, size_in, size_out, bias=True, enable_rlrr=False):
        super(MoPDOLinear, self).__init__()
        self.enable_rlrr = enable_rlrr
        self.size_in = size_in
        self.size_out = size_out
        self.has_bias = bias
        self.mlp = nn.Linear(size_in, size_out, bias=bias)
        if self.enable_rlrr:
            self.scale_col = nn.Parameter(torch.empty(1, self.size_in), requires_grad=True)
            self.scale_line = nn.Parameter(torch.empty(self.size_out, 1), requires_grad=True)
            self.shift_bias = nn.Parameter(torch.empty(1, self.size_out), requires_grad=True)

            # SSF
            # nn.init.normal_(self.scale_line, mean=1, std=0.02)
            # nn.init.normal_(self.scale_col, mean=1, std=0.02)
            # nn.init.normal_(self.shift_bias, mean=0, std=0.02)

            # our
            nn.init.kaiming_uniform_(self.scale_col)
            nn.init.kaiming_uniform_(self.scale_line)
            nn.init.zeros_(self.shift_bias)

        self._frozen_param()

    def _frozen_param(self):
        for param in self.mlp.parameters():
            param.requires_grad = False

    def forward(self, x):           
        if self.enable_rlrr:
            weight = (self.mlp.weight * self.scale_col * self.scale_line) + self.mlp.weight
            bias = self.mlp.bias + self.shift_bias
            return F.linear(x, weight, bias)
        else:
            return self.mlp(x)


class MoPDOAttention(nn.Module):
    def __init__(self, config, vis, enable_rlrr=True):
        super(MoPDOAttention, self).__init__()
        
        # new
        enable_rlrr = False
        
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = MoPDOLinear(config.hidden_size, self.all_head_size, bias=True, enable_rlrr=False) #True) #=enable_rlrr)
        self.key = MoPDOLinear(config.hidden_size, self.all_head_size, bias=True, enable_rlrr=False)# True) #=enable_rlrr)
        self.value = MoPDOLinear(config.hidden_size, self.all_head_size, bias=True, enable_rlrr=True) #enable_rlrr)
        self.out = MoPDOLinear(config.hidden_size, config.hidden_size, bias=True, enable_rlrr=True) #enable_rlrr)
        self.attn_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
    
        self.enable_rlrr = False
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output  # , weights


class MoPDOMLP(nn.Module):
    def __init__(self, config, enable_rlrr=True):
        super(MoPDOMLP, self).__init__()
        self.fc1 = MoPDOLinear(config.hidden_size, config.transformer["mlp_dim"], bias=True, enable_rlrr=enable_rlrr)
        self.fc2 = MoPDOLinear(config.transformer["mlp_dim"], config.hidden_size, bias=True, enable_rlrr=enable_rlrr)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        x = self.act(self.fc1(x))
        if self.training:
            x = self.dropout(x)
        x = self.fc2(x)
        if self.training:
            x = self.dropout(x)
        return x
    
def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))

    nn.init.normal_(scale, mean=1, std=.02)
    nn.init.normal_(shift, std=.02)

    return scale, shift


def ssf_ada(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

class MoPDOBlock(nn.Module):
    def __init__(self, config, vis, drop_path=0.0, enable_rlrr=True, shared_pde_parameters=None):
        super(MoPDOBlock, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = MoPDOMLP(config, enable_rlrr=enable_rlrr)
        self.attn = MoPDOAttention(config, vis, enable_rlrr=enable_rlrr)
        self.enable_rlrr = enable_rlrr
        self.drop_path1 = nn.Dropout(p=drop_path)
        self.drop_path2 = nn.Dropout(p=drop_path)
        self.vis_save_dir = None
        self.vis_block_idx = None
        self.hco_enabled = bool(enable_rlrr)
        if self.enable_rlrr:
            hco_kwargs = {}
            if shared_pde_parameters is not None:
                hco_kwargs = dict(
                    heat_k_init_value=shared_pde_parameters[0],
                    wave_k_init_value=shared_pde_parameters[1],
                    possion_k_init_value=shared_pde_parameters[2],
                )
            self.hco = HCO(
                res=14,
                dim=self.hidden_size,
                num_heads=config.transformer["num_heads"],
                **hco_kwargs,
            )
            self.ssf_scale_ffn_norm, self.ssf_shift_ffn_norm = init_ssf_scale_shift(self.hidden_size)

    def set_hco_enabled(self, enabled=True):
        """Enable/disable only the spectral HCO path for an ablation.

        The RLRR/SSF affine path and attention remain active when HCO is
        disabled, so this isolates the PDE operator instead of turning off all
        MoPDO adapters.
        """
        self.hco_enabled = bool(enabled)
        if not self.hco_enabled and hasattr(self, "hco"):
            for parameter in self.hco.parameters():
                parameter.requires_grad_(False)
        return self

    @torch.no_grad()
    def _tensor_to_224_map(self, q_tensor):
        """
        q_tensor: [B, N, C], usually [1, 197, 768].
        Return: [B, 1, 224, 224], min-max normalized.
        """
        q = q_tensor.detach()

        # Remove CLS token if possible.
        if q.shape[1] > 1:
            spatial_tokens = q.shape[1] - 1
            h = int(math.sqrt(spatial_tokens))
            if h * h == spatial_tokens:
                q = q[:, 1:, :]

        B, N, C = q.shape
        h = int(math.sqrt(N))
        assert h * h == N, f"Cannot reshape N={N} tokens to square feature map."

        # Channel-averaged magnitude map: [B, N] -> [B, 1, H, W]
        q_map = q.abs().mean(dim=-1).reshape(B, 1, h, h)

        # Resize to 224x224
        q_map = F.interpolate(q_map, size=(224, 224), mode="bilinear", align_corners=False)

        # Min-max normalize each sample
        q_min = q_map.flatten(1).min(dim=1)[0].view(B, 1, 1, 1)
        q_max = q_map.flatten(1).max(dim=1)[0].view(B, 1, 1, 1)
        q_map = (q_map - q_min) / (q_max - q_min + 1e-6)

        return q_map

    @torch.no_grad()
    def _save_q_visualization(self, q_base, q_hco, q_diff):
        """
        Save q_base / q_hco / q_diff tensors and their 224x224 normalized maps/images.
        """
        if getattr(self, "vis_save_dir", None) is None:
            return

        block_idx = getattr(self, "vis_block_idx", None)
        if block_idx is None:
            block_idx = "unknown"

        block_dir = os.path.join(self.vis_save_dir, f"block_{block_idx}")
        os.makedirs(block_dir, exist_ok=True)

        # Save raw tensors: [B, N, C]
        torch.save(q_base.detach().cpu(), os.path.join(block_dir, "q_base.pt"))
        torch.save(q_hco.detach().cpu(), os.path.join(block_dir, "q_hco.pt"))
        torch.save(q_diff.detach().cpu(), os.path.join(block_dir, "q_diff.pt"))

        # Save normalized 224x224 maps: [B, 1, 224, 224]
        q_base_map = self._tensor_to_224_map(q_base)
        q_hco_map = self._tensor_to_224_map(q_hco)
        q_diff_map = self._tensor_to_224_map(q_diff)

        torch.save(q_base_map.cpu(), os.path.join(block_dir, "q_base_map_224.pt"))
        torch.save(q_hco_map.cpu(), os.path.join(block_dir, "q_hco_map_224.pt"))
        torch.save(q_diff_map.cpu(), os.path.join(block_dir, "q_diff_map_224.pt"))

        save_image(q_base_map.cpu(), os.path.join(block_dir, "q_base_map_224.png"))
        save_image(q_hco_map.cpu(), os.path.join(block_dir, "q_hco_map_224.png"))
        save_image(q_diff_map.cpu(), os.path.join(block_dir, "q_diff_map_224.png"))

    def to(self, *args, **kwargs):
        
        # ���ø���� to ����
        super().to(*args, **kwargs)

        # ȷ������ HCO ģ��Ҳ���ƶ���ָ���豸
        for module in self.children():
            if isinstance(module, nn.Module):
                module.to(*args, **kwargs)

        return self

    def forward(self, x, is_training=False, epoch=0, max_epoch=110, entropy_scale=0.1):
        if self.enable_rlrr:
            if False:
                x = x + self.drop_path1(self.attn(self.attention_norm(self.hco(x))))
                x = x + self.drop_path2(self.ffn(ssf_ada(self.ffn_norm(x), self.ssf_scale_ffn_norm, self.ssf_shift_ffn_norm)))
            # x after HCO; for the no-HCO ablation preserve the surrounding
            # RLRR/SSF path and use the identity operator.
            if self.hco_enabled:
                hco_out = self.hco(x, is_training=is_training, epoch=epoch, max_epoch=max_epoch, entropy_scale=entropy_scale)
                if is_training:
                    x_hco, loss_entropy = hco_out
                else:
                    x_hco = hco_out
                    loss_entropy = None
            else:
                x_hco = x
                loss_entropy = x.new_zeros(()) if is_training else None
            x_hco_norm = self.attention_norm(x_hco)

            # Q probes are analysis-only. Avoid adding two query projections to
            # every normal training/inference pass when visualization is off.
            if self.vis_save_dir is not None:
                with torch.no_grad():
                    q_base = self.attn.query(self.attention_norm(x))
                    q_hco = self.attn.query(x_hco_norm)
                    self._save_q_visualization(q_base, q_hco, q_hco - q_base)

            x = x + self.drop_path1(self.attn(x_hco_norm))
            x = x + self.drop_path2(
                self.ffn(
                    ssf_ada(
                        self.ffn_norm(x),
                        self.ssf_scale_ffn_norm,
                        self.ssf_shift_ffn_norm
                    )
                )
            )
            if is_training:
                return x, loss_entropy
        else:
            x = x + self.drop_path1(self.attn(self.attention_norm(x)))
            x = x + self.drop_path2(self.ffn(self.ffn_norm(x)))
            if is_training:
                return x, x.new_zeros(())
        return x

    def _fc_load_weight(self, ROOT, Key, Weights, unit):
        mat_weights = Weights[ROOT + '/' + Key + '/' + "kernel"]
        mat_bias = Weights[ROOT + '/' + Key + '/' + "bias"]
        if Key == ATTENTION_OUT:
            mat_weights = mat_weights.reshape(-1, mat_weights.shape[-1])
        else:
            mat_weights = mat_weights.reshape(mat_weights.shape[0], -1)

        unit.mlp.weight.copy_(np2th(mat_weights).t())
        unit.mlp.bias.copy_(np2th(mat_bias).view(-1))

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            self._fc_load_weight(ROOT, ATTENTION_Q, weights, self.attn.query)
            self._fc_load_weight(ROOT, ATTENTION_K, weights, self.attn.key)
            self._fc_load_weight(ROOT, ATTENTION_V, weights, self.attn.value)
            self._fc_load_weight(ROOT, ATTENTION_OUT, weights, self.attn.out)
            self._fc_load_weight(ROOT, FC_0, weights, self.ffn.fc1)
            self._fc_load_weight(ROOT, FC_1, weights, self.ffn.fc2)

            self.attention_norm.weight.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + '/' + "scale"]))
            self.attention_norm.bias.copy_(np2th(weights[ROOT + '/' + ATTENTION_NORM + '/' + "bias"]))
            self.ffn_norm.weight.copy_(np2th(weights[ROOT + '/' + MLP_NORM + '/' + "scale"]))
            self.ffn_norm.bias.copy_(np2th(weights[ROOT + '/' + MLP_NORM + '/' + "bias"]))


class MoPDOEncoder(nn.Module):
    def __init__(self, config, vis, drop_path=0.0, enable_rlrr=True):
        super(MoPDOEncoder, self).__init__()
        self.vis = vis
        self.enable_rlrr = enable_rlrr
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.num_blocks = config.transformer["num_layers"]
        shared_pde_parameters = None
        if self.enable_rlrr:
            coefficient_shape = (14, 14, config.transformer["num_heads"])
            shared_pde_parameters = tuple(
                nn.Parameter(torch.ones(coefficient_shape)) for _ in range(3)
            )
        # fellow SSF
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]  # stochastic depth decay rule
        for i in range(config.transformer["num_layers"]):
            self.layer.append(
                MoPDOBlock(
                    config,
                    vis,
                    drop_path=dpr[i],
                    enable_rlrr=enable_rlrr,
                    shared_pde_parameters=shared_pde_parameters,
                )
            )
        
        if self.enable_rlrr:
            self.ssf_scale_enc_norm, self.ssf_shift_enc_norm = init_ssf_scale_shift(config.hidden_size)
    
    def to(self, *args, **kwargs):
    
        # ���ø���� to ����
        super().to(*args, **kwargs)

        # �ݹ���ÿ����ģ��
        for module in self.children():
            if isinstance(module, nn.ModuleList):
                # �����ģ���� ModuleList���������е�ÿ��ģ��
                for sub_module in module:
                    if isinstance(sub_module, HCO):
                        sub_module.to(*args, **kwargs)  # ȷ�� HCO �ƶ�����ȷ�豸
                    else:
                        sub_module.to(*args, **kwargs)  # ������ģ��Ҳ�ƶ����豸

        return self

    def forward(self, hidden_states, is_training=False, epoch=0, max_epoch=110, entropy_scale=0.1):
        loss_entropy = None
        for layer_block in self.layer:
            if is_training:
                hidden_states, entropy = layer_block(
                    hidden_states,
                    is_training=is_training,
                    epoch=epoch,
                    max_epoch=max_epoch,
                    entropy_scale=entropy_scale,
                )
                if loss_entropy is None:
                    loss_entropy = entropy
                else:
                    loss_entropy = loss_entropy + entropy
            else:
                hidden_states = layer_block(hidden_states)
                
        encoded = self.encoder_norm(hidden_states)
        if self.enable_rlrr: 
            encoded = ssf_ada(encoded, self.ssf_scale_enc_norm, self.ssf_shift_enc_norm)
        if is_training:
            if loss_entropy is None:
                loss_entropy = encoded.new_zeros(())
            return encoded, loss_entropy
        return encoded


class MoPDOEmbeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3, enable_rlrr=True):
        super(MoPDOEmbeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.hybrid = False

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches + 1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = nn.Dropout(config.transformer["dropout_rate"])
        self.enable_rlrr = enable_rlrr
        if self.enable_rlrr:
            self.ssf_scale_patch, self.ssf_shift_patch = init_ssf_scale_shift(config.hidden_size)
            
    def to(self, *args, **kwargs):
        
        # ���ø���� to ����
        super().to(*args, **kwargs)

        # ȷ������ HCO ģ��Ҳ���ƶ���ָ���豸
        for module in self.children():
            if isinstance(module, nn.Module):
                module.to(*args, **kwargs)

        return self

    def forward(self, x):
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        if self.enable_rlrr:  
            x = ssf_ada(x, self.ssf_scale_patch, self.ssf_shift_patch)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        
        return embeddings


class MoPDOTransformer(nn.Module):
    def __init__(self, config, img_size, vis, drop_path=0.0, enable_rlrr=True):
        super(MoPDOTransformer, self).__init__()
        self.embeddings = MoPDOEmbeddings(config, img_size=img_size)
        self.encoder = MoPDOEncoder(config, vis, drop_path=drop_path, enable_rlrr=enable_rlrr)
        self._frozen_param()
        self.enable_rlrr = enable_rlrr
        
    def to(self, *args, **kwargs):
        
        # ���ø���� to ����
        super().to(*args, **kwargs)

        # ȷ������ HCO ģ��Ҳ���ƶ���ָ���豸
        for module in self.children():
            if isinstance(module, nn.Module):
                module.to(*args, **kwargs)

        return self

    def _frozen_param(self):
        for param in self.embeddings.parameters():
            param.requires_grad = False


    def forward(self, input_ids, is_training=False, epoch=0, max_epoch=110, entropy_scale=0.1):
        embedding_output = self.embeddings(input_ids)
        if is_training:
            encoded, loss_entropy = self.encoder(
                embedding_output,
                is_training=is_training,
                epoch=epoch,
                max_epoch=max_epoch,
                entropy_scale=entropy_scale,
            )
            return encoded, loss_entropy
        encoded = self.encoder(embedding_output)
        return encoded


class MoPDOVisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, enable_rlrr=True, drop_path=0.0):
        super(MoPDOVisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = MoPDOTransformer(config, img_size, vis, drop_path=drop_path, enable_rlrr=enable_rlrr)
        self.head = nn.Linear(config.hidden_size, num_classes)
        self.loss_fct = nn.CrossEntropyLoss()

    def to(self, *args, **kwargs):
    
        # ���ø���� to ����
        super().to(*args, **kwargs)

        # ȷ������ HCO ģ��Ҳ���ƶ���ָ���豸
        for module in self.children():
            if isinstance(module, nn.Module):
                module.to(*args, **kwargs)

        return self

    def get_parameters(self, lr, weight_decay):
        wd_params = []
        no_wd_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        params = [
            {"params": wd_params, "lr": lr, "weight_decay": weight_decay},
            {"params": no_wd_params, "lr": lr, "weight_decay": 0.}
        ]

        return params

    def forward(self, x, labels=None, is_training=False, epoch=0, max_epoch=110, entropy_scale=0.1):
        if is_training:
            x, loss_entropy = self.transformer(
                x,
                is_training=is_training,
                epoch=epoch,
                max_epoch=max_epoch,
                entropy_scale=entropy_scale,
            )
        else:
            x = self.transformer(x)

        logits = self.head(x[:, 0])
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            if is_training:
                return loss, loss_entropy
            return loss
        if is_training:
            return logits, loss_entropy
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
if __name__ == '__main__':
    config = CONFIGS['ViT-B_16']
    model = MoPDOVisionTransformer(config, 224, zero_head=False, num_classes=1000)
    params = sum(p.numel() for p in model.parameters())
    print(f"Built {model.__class__.__name__} with {params:,} parameters")
