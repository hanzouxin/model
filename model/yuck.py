# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class BPCompatibleBipartiteMP(nn.Module):
#     """
#     BP-friendly sparse bipartite message passing
#     - affinity S: cosine-like (normalized) for stability
#     - message aggregation: use raw features (keep magnitude for hash head)
#     - output: residual + LN (NO final L2 normalize)
#     """
#     def __init__(self, dim=512, tau=0.07, topk=16, drop=0.0, use_proj=True,
#                  detach_affinity=True, alpha=1.0, beta=0.1, use_bottom_k=False):
#         super().__init__()
#         self.tau = tau
#         self.topk = topk
#         self.drop = nn.Dropout(drop)
#         self.detach_affinity = detach_affinity
#         self.alpha = alpha
#         self.beta = beta
#         self.use_bottom_k = use_bottom_k  # Whether to use bottom-k for rejection of bad neighbors

#         self.proj_i = nn.Linear(dim, dim, bias=False) if use_proj else nn.Identity()
#         self.proj_t = nn.Linear(dim, dim, bias=False) if use_proj else nn.Identity()
#         self.ln_i = nn.LayerNorm(dim)
#         self.ln_t = nn.LayerNorm(dim)

#     def _row_topk_softmax(self, S: torch.Tensor, k: int):
#         Bi, Bt = S.shape
#         if k <= 0 or k >= Bt:
#             return torch.softmax(S, dim=1)

#         neg_large = -1e4 if S.dtype in (torch.float16, torch.bfloat16) else -1e9
#         idx = S.topk(k, dim=1).indices
#         mask = torch.zeros_like(S, dtype=torch.bool)
#         mask.scatter_(1, idx, True)
#         S_masked = S.masked_fill(~mask, neg_large)
#         return torch.softmax(S_masked, dim=1)

#     def forward(self, f_img: torch.Tensor, f_txt: torch.Tensor, alpha: float = None):
#         """
#         f_img: (Bi,D) raw features
#         f_txt: (Bt,D) raw features
#         """
#         if alpha is None:
#             alpha = self.alpha

#         # 1) affinity 用 normalize 的特征算（稳定）
#         fi_n = F.normalize(f_img, dim=-1)  # 用归一化特征计算相似度
#         ft_n = F.normalize(f_txt, dim=-1)  # 用归一化特征计算相似度
#         S = (fi_n @ ft_n.t()) / self.tau  # (Bi,Bt)
#         if self.detach_affinity:
#             S = S.detach()  # 防止相似度矩阵的梯度传播影响 BP 损失

#         # 2) image <- texts
#         A_it = self._row_topk_softmax(S, self.topk)
#         A_it = self.drop(A_it)
#         msg_i = A_it @ self.proj_t(f_txt)  # 注意：这里用原始特征

#         # 3) text <- images
#         A_ti = self._row_topk_softmax(S.t(), self.topk)
#         A_ti = self.drop(A_ti)
#         msg_t = A_ti @ self.proj_i(f_img)  # 注意：这里用原始特征

#         # 4) 排斥消息：bottom-k（用于防止硬负样本传播） - 可选
#         if self.use_bottom_k:
#             # 对相似度矩阵 S 进行 bottom-k 操作，排斥最不相似的邻居
#             A_it_neg = self._row_topk_softmax(-S, self.topk)  # 对负相似度做 top-k
#             msg_i_neg = A_it_neg @ self.proj_t(f_txt)  # 消息传递
#             msg_i = msg_i - self.beta * msg_i_neg  # 加权调整

#             A_ti_neg = self._row_topk_softmax(-S.t(), self.topk)  # 对负相似度做 top-k
#             msg_t_neg = A_ti_neg @ self.proj_i(f_img)  # 消息传递
#             msg_t = msg_t - self.beta * msg_t_neg  # 加权调整

#         # 5) 输出不做 L2 normalize，交给 hash head 自己决定尺度
#         f_img2 = self.ln_i(f_img + alpha * msg_i)
#         f_txt2 = self.ln_t(f_txt + alpha * msg_t)

#         return f_img2, f_txt2
    

# class PairwiseCrossGatingFusion(nn.Module):
#     """
#     Pairwise (per-sample) cross-modal fusion, NO batch mixing.
#     Input:  f_img, f_txt  (B,D)
#     Output: f_img', f_txt' (B,D)
#     """
#     def __init__(self, dim=512, alpha=0.5, drop=0.0, hidden=1024,
#                  detach_gate=False, detach_msg=False):
#         super().__init__()
#         self.alpha = alpha
#         self.drop = nn.Dropout(drop)
#         self.detach_gate = detach_gate
#         self.detach_msg = detach_msg

#         # cross messages
#         self.t2i = nn.Linear(dim, dim, bias=False)
#         self.i2t = nn.Linear(dim, dim, bias=False)

#         # gates: use rich pairwise interactions (cat, prod, diff)
#         gate_in = dim * 4
#         self.gate_i = nn.Sequential(
#             nn.Linear(gate_in, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, dim),
#             nn.Sigmoid()
#         )
#         self.gate_t = nn.Sequential(
#             nn.Linear(gate_in, hidden),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, dim),
#             nn.Sigmoid()
#         )

#         self.ln_i = nn.LayerNorm(dim)
#         self.ln_t = nn.LayerNorm(dim)

#     def forward(self, f_img, f_txt, alpha=None):
#         if alpha is None:
#             alpha = self.alpha

#         # messages (keep raw magnitude; BP-friendly)
#         msg_i = self.t2i(f_txt)
#         msg_t = self.i2t(f_img)

#         if self.detach_msg:
#             msg_i = msg_i.detach()
#             msg_t = msg_t.detach()

#         # gates computed from pairwise interactions
#         prod = f_img * f_txt
#         diff_it = f_img - f_txt
#         diff_ti = -diff_it

#         gi_in = torch.cat([f_img, f_txt, prod, diff_it], dim=-1)
#         gt_in = torch.cat([f_txt, f_img, prod, diff_ti], dim=-1)

#         g_i = self.gate_i(gi_in)
#         g_t = self.gate_t(gt_in)

#         if self.detach_gate:
#             g_i = g_i.detach()
#             g_t = g_t.detach()

#         msg_i = self.drop(g_i * msg_i)
#         msg_t = self.drop(g_t * msg_t)

#         out_i = self.ln_i(f_img + alpha * msg_i)
#         out_t = self.ln_t(f_txt + alpha * msg_t)
#         return out_i, out_t


# class BPThenPairFusion(nn.Module):
#     def __init__(self, dim=512):
#         super().__init__()
#         self.mp = BPCompatibleBipartiteMP(dim=dim, tau=0.07, topk=16, drop=0.0,
#                                           use_proj=True, detach_affinity=True,
#                                           alpha=1.0, beta=0.1, use_bottom_k=False)
#         self.fuse = PairwiseCrossGatingFusion(dim=dim, alpha=0.5, drop=0.1,
#                                               detach_gate=False, detach_msg=False)

#     def forward(self, f_img, f_txt):
#         f_img2, f_txt2 = self.mp(f_img, f_txt)     # cross: 你现有稀疏二部图
#         f_img3, f_txt3 = self.fuse(f_img2, f_txt2) # pairwise: 只做配对融合，不跨 batch
#         return f_img3, f_txt3



# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BPCompatibleBipartiteMP(nn.Module):
#     def __init__(self, dim=512, tau=0.07, topk=16, drop=0.0, use_proj=True,
#                  detach_affinity=True, alpha=1.0, beta=0.1, use_bottom_k=False):
#         super().__init__()
#         self.tau = tau
#         self.topk = topk
#         self.drop = nn.Dropout(drop)
#         self.detach_affinity = detach_affinity
#         self.alpha = alpha
#         self.beta = beta
#         self.use_bottom_k = use_bottom_k

#         self.proj_i = nn.Linear(dim, dim, bias=False) if use_proj else nn.Identity()
#         self.proj_t = nn.Linear(dim, dim, bias=False) if use_proj else nn.Identity()
#         self.ln_i = nn.LayerNorm(dim)
#         self.ln_t = nn.LayerNorm(dim)

#     def _row_topk_softmax(self, S: torch.Tensor, k: int):
#         Bi, Bt = S.shape
#         if k <= 0 or k >= Bt:
#             return torch.softmax(S, dim=1)

#         neg_large = -1e4 if S.dtype in (torch.float16, torch.bfloat16) else -1e9
#         idx = S.topk(k, dim=1).indices
#         mask = torch.zeros_like(S, dtype=torch.bool)
#         mask.scatter_(1, idx, True)
#         S_masked = S.masked_fill(~mask, neg_large)
#         return torch.softmax(S_masked, dim=1)

#     def forward(self, f_img: torch.Tensor, f_txt: torch.Tensor, alpha: float = None, return_msg: bool = False):
#         if alpha is None:
#             alpha = self.alpha

#         fi_n = F.normalize(f_img, dim=-1)
#         ft_n = F.normalize(f_txt, dim=-1)
#         S = (fi_n @ ft_n.t()) / self.tau
#         if self.detach_affinity:
#             S = S.detach()

#         A_it = self._row_topk_softmax(S, self.topk)
#         A_it = self.drop(A_it)
#         msg_i = A_it @ self.proj_t(f_txt)   # raw f_txt 聚合过来

#         A_ti = self._row_topk_softmax(S.t(), self.topk)
#         A_ti = self.drop(A_ti)
#         msg_t = A_ti @ self.proj_i(f_img)   # raw f_img 聚合过来

#         if self.use_bottom_k:
#             A_it_neg = self._row_topk_softmax(-S, self.topk)
#             msg_i_neg = A_it_neg @ self.proj_t(f_txt)
#             msg_i = msg_i - self.beta * msg_i_neg

#             A_ti_neg = self._row_topk_softmax(-S.t(), self.topk)
#             msg_t_neg = A_ti_neg @ self.proj_i(f_img)
#             msg_t = msg_t - self.beta * msg_t_neg

#         f_img2 = self.ln_i(f_img + alpha * msg_i)
#         f_txt2 = self.ln_t(f_txt + alpha * msg_t)

#         if return_msg:
#             return f_img2, f_txt2, msg_i, msg_t
#         return f_img2, f_txt2



# class KANLinear(nn.Module):
#     """
#     Piecewise-linear spline KAN layer.
#     y_k = sum_j g_{k,j}(x_j) + b_k
#     g_{k,j} is defined on a fixed grid with learnable values (coeffs).
#     """
#     def __init__(self, in_dim, out_dim, grid_size=8, bias=True):
#         super().__init__()
#         assert grid_size >= 2
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.M = grid_size

#         # coeff: (out, in, M)
#         self.coeff = nn.Parameter(0.01 * torch.randn(out_dim, in_dim, grid_size))
#         self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

#         # learn per-input scale to keep x in a good range
#         self.log_scale = nn.Parameter(torch.zeros(in_dim))

#         # fixed grid in [-1,1]
#         self.register_buffer("grid", torch.linspace(-1.0, 1.0, grid_size), persistent=False)
#         self.h = 2.0 / (grid_size - 1)

#     def forward(self, x):
#         # x: (B, in)
#         B, Din = x.shape
#         assert Din == self.in_dim

#         scale = F.softplus(self.log_scale) + 1e-3     # (in,)
#         x = torch.tanh(x / scale)                     # map to [-1,1]

#         # position in grid
#         t = (x + 1.0) / self.h                        # in [0, M-1]
#         i0 = torch.floor(t).long().clamp(0, self.M - 2)  # (B,in)
#         frac = (t - i0.float()).clamp(0.0, 1.0)          # (B,in)

#         # gather c0, c1 along grid dim
#         # coeff_exp: (B,out,in,M)
#         coeff_exp = self.coeff.unsqueeze(0).expand(B, -1, -1, -1)

#         idx0 = i0.unsqueeze(1).expand(B, self.out_dim, self.in_dim).unsqueeze(-1)           # (B,out,in,1)
#         idx1 = (i0 + 1).unsqueeze(1).expand(B, self.out_dim, self.in_dim).unsqueeze(-1)     # (B,out,in,1)

#         c0 = torch.gather(coeff_exp, 3, idx0).squeeze(-1)   # (B,out,in)
#         c1 = torch.gather(coeff_exp, 3, idx1).squeeze(-1)   # (B,out,in)

#         w = frac.unsqueeze(1)                                # (B,1,in)
#         g = (1.0 - w) * c0 + w * c1                           # (B,out,in)

#         y = g.sum(dim=2)                                      # (B,out)
#         if self.bias is not None:
#             y = y + self.bias
#         return y


# class KANMsgFusion(nn.Module):
#     """
#     Fuse feature with its cross-modal message using KAN.
#     f' = LN(f + alpha * DeltaKAN([f, msg, f*msg, f-msg]))
#     """
#     def __init__(self, dim=512, hidden=256, grid_size=8, alpha=0.3, drop=0.0):
#         super().__init__()
#         self.alpha = alpha
#         self.drop = nn.Dropout(drop)

#         self.pre = nn.Linear(dim * 4, hidden)
#         self.kan1 = KANLinear(hidden, hidden, grid_size=grid_size)
#         self.kan2 = KANLinear(hidden, hidden, grid_size=grid_size)
#         self.out = nn.Linear(hidden, dim)

#         self.ln = nn.LayerNorm(dim)

#     def forward(self, f, msg, alpha=None):
#         if alpha is None:
#             alpha = self.alpha
#         x = torch.cat([f, msg, f * msg, f - msg], dim=-1)  # (B,4D)
#         h = F.gelu(self.pre(x))
#         h = F.gelu(self.kan1(h))
#         h = self.kan2(h)
#         delta = self.drop(self.out(h))
#         return self.ln(f + alpha * delta)
# class BPThenKANFusion(nn.Module):
#     def __init__(self, dim=512,
#                  # bipartite
#                  tau=0.07, topk=16, drop=0.0, detach_affinity=True,
#                  alpha_mp=1.0, beta=0.1, use_bottom_k=False,
#                  # KAN
#                  hidden=256, grid_size=8, alpha_kan=0.3, drop_kan=0.1):
#         super().__init__()
#         self.mp = BPCompatibleBipartiteMP(
#             dim=dim, tau=tau, topk=topk, drop=drop, use_proj=True,
#             detach_affinity=detach_affinity, alpha=alpha_mp, beta=beta, use_bottom_k=use_bottom_k
#         )
#         self.fuse_img = KANMsgFusion(dim=dim, hidden=hidden, grid_size=grid_size, alpha=alpha_kan, drop=drop_kan)
#         self.fuse_txt = KANMsgFusion(dim=dim, hidden=hidden, grid_size=grid_size, alpha=alpha_kan, drop=drop_kan)

#     def forward(self, f_img, f_txt):
#         f_img2, f_txt2, msg_i, msg_t = self.mp(f_img, f_txt, return_msg=True)
#         # 用“跨模态聚合消息”做 KAN 融合（不会乱融合错配样本）
#         f_img3 = self.fuse_img(f_img2, msg_i)
#         f_txt3 = self.fuse_txt(f_txt2, msg_t)
#         return f_img3, f_txt3


# -*- coding: utf-8 -*-
# All-in-one: BPCompatibleBipartiteMP (returns messages) + KAN (KANLinear/KAN) + BPThenKAN fusion
# You can paste this into one .py file and import BPThenKAN.



# # =========================================================
# # 1) BP-friendly Sparse Bipartite Message Passing (returns msg)
# # =========================================================
# class BPCompatibleBipartiteMP(nn.Module):
#     """
#     BP-friendly sparse bipartite message passing
#     - affinity S: cosine-like (normalized) for stability
#     - message aggregation: use raw features (keep magnitude for hash head)
#     - output: residual + LN (NO final L2 normalize)
#     - optional bottom-k rejection
#     """
#     def __init__(self,
#                  dim=512,
#                  tau=0.07,
#                  topk=16,
#                  drop=0.0,
#                  use_proj=True,
#                  detach_affinity=True,
#                  alpha=1.0,
#                  beta=0.1,
#                  use_bottom_k=False):
#         super().__init__()
#         self.tau = tau
#         self.topk = topk
#         self.drop = nn.Dropout(drop)
#         self.detach_affinity = detach_affinity
#         self.alpha = alpha
#         self.beta = beta
#         self.use_bottom_k = use_bottom_k

#         self.proj_i = nn.Linear(dim, dim, bias=False) if use_proj else nn.Identity()
#         self.proj_t = nn.Linear(dim, dim, bias=False) if use_proj else nn.Identity()
#         self.ln_i = nn.LayerNorm(dim)
#         self.ln_t = nn.LayerNorm(dim)

#     def _row_topk_softmax(self, S: torch.Tensor, k: int):
#         # S: (Bi,Bt)
#         Bi, Bt = S.shape
#         if k <= 0 or k >= Bt:
#             return torch.softmax(S, dim=1)

#         neg_large = -1e4 if S.dtype in (torch.float16, torch.bfloat16) else -1e9
#         idx = S.topk(k, dim=1).indices               # (Bi,k)
#         mask = torch.zeros_like(S, dtype=torch.bool) # (Bi,Bt)
#         mask.scatter_(1, idx, True)

#         S_masked = S.masked_fill(~mask, neg_large)
#         return torch.softmax(S_masked, dim=1)

#     def forward(self,
#                 f_img: torch.Tensor,
#                 f_txt: torch.Tensor,
#                 alpha: float = None,
#                 return_msg: bool = False):
#         """
#         f_img: (Bi,D) raw features
#         f_txt: (Bt,D) raw features
#         """
#         if alpha is None:
#             alpha = self.alpha

#         # 1) affinity from normalized features (stable)
#         fi_n = F.normalize(f_img, dim=-1)
#         ft_n = F.normalize(f_txt, dim=-1)
#         S = (fi_n @ ft_n.t()) / self.tau  # (Bi,Bt)

#         if self.detach_affinity:
#             S = S.detach()

#         # 2) image <- texts
#         A_it = self._row_topk_softmax(S, self.topk)  # (Bi,Bt)
#         A_it = self.drop(A_it)
#         msg_i = A_it @ self.proj_t(f_txt)            # (Bi,D)  raw f_txt values

#         # 3) text <- images
#         A_ti = self._row_topk_softmax(S.t(), self.topk)  # (Bt,Bi)
#         A_ti = self.drop(A_ti)
#         msg_t = A_ti @ self.proj_i(f_img)                # (Bt,D)  raw f_img values

#         # 4) optional bottom-k rejection
#         if self.use_bottom_k:
#             # bottom-k on S = top-k on (-S)
#             A_it_neg = self._row_topk_softmax(-S, self.topk)
#             msg_i_neg = A_it_neg @ self.proj_t(f_txt)
#             msg_i = msg_i - self.beta * msg_i_neg

#             A_ti_neg = self._row_topk_softmax(-S.t(), self.topk)
#             msg_t_neg = A_ti_neg @ self.proj_i(f_img)
#             msg_t = msg_t - self.beta * msg_t_neg

#         # 5) residual + LN (no L2 norm)
#         f_img2 = self.ln_i(f_img + alpha * msg_i)
#         f_txt2 = self.ln_t(f_txt + alpha * msg_t)

#         if return_msg:
#             return f_img2, f_txt2, msg_i, msg_t
#         return f_img2, f_txt2


# # =========================================================
# # 2) KAN implementation (KANLinear + KAN)
# #    (copied/adapted from your reference; kept consistent & runnable)
# # =========================================================
# class KANLinear(nn.Module):
#     def __init__(
#         self,
#         in_features,
#         out_features,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         enable_standalone_scale_spline=True,
#         base_activation=nn.SiLU,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.grid_size = grid_size
#         self.spline_order = spline_order

#         h = (grid_range[1] - grid_range[0]) / grid_size
#         grid = (
#             (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
#             .expand(in_features, -1)
#             .contiguous()
#         )
#         self.register_buffer("grid", grid)  # (in_features, grid_size + 2*spline_order + 1)

#         self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
#         self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
#         if enable_standalone_scale_spline:
#             self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
#         else:
#             self.spline_scaler = None

#         self.scale_noise = scale_noise
#         self.scale_base = scale_base
#         self.scale_spline = scale_spline
#         self.enable_standalone_scale_spline = enable_standalone_scale_spline
#         self.base_activation = base_activation()
#         self.grid_eps = grid_eps

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
#         with torch.no_grad():
#             noise = (
#                 (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
#                 * self.scale_noise
#                 / self.grid_size
#             )
#             self.spline_weight.data.copy_(
#                 (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
#                 * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
#             )
#             if self.enable_standalone_scale_spline:
#                 nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

#     def b_splines(self, x: torch.Tensor):
#         """
#         x: (B, in_features)
#         return: (B, in_features, grid_size + spline_order)
#         """
#         assert x.dim() == 2 and x.size(1) == self.in_features

#         grid: torch.Tensor = self.grid  # (in_features, grid_size + 2*spline_order + 1)
#         x = x.unsqueeze(-1)            # (B, in, 1)

#         bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # (B,in,grid+2*order)
#         for k in range(1, self.spline_order + 1):
#             bases = (
#                 (x - grid[:, : -(k + 1)])
#                 / (grid[:, k:-1] - grid[:, : -(k + 1)])
#                 * bases[:, :, :-1]
#             ) + (
#                 (grid[:, k + 1 :] - x)
#                 / (grid[:, k + 1 :] - grid[:, 1:(-k)])
#                 * bases[:, :, 1:]
#             )

#         assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
#         return bases.contiguous()

#     def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
#         """
#         x: (B, in_features)
#         y: (B, in_features, out_features)
#         return: (out_features, in_features, grid_size + spline_order)
#         """
#         assert x.dim() == 2 and x.size(1) == self.in_features
#         assert y.size() == (x.size(0), self.in_features, self.out_features)

#         A = self.b_splines(x).transpose(0, 1)  # (in, B, coeff)
#         B = y.transpose(0, 1)                  # (in, B, out)
#         solution = torch.linalg.lstsq(A, B).solution  # (in, coeff, out)
#         result = solution.permute(2, 0, 1)             # (out, in, coeff)

#         assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order)
#         return result.contiguous()

#     @property
#     def scaled_spline_weight(self):
#         if self.enable_standalone_scale_spline:
#             return self.spline_weight * self.spline_scaler.unsqueeze(-1)
#         return self.spline_weight

#     def forward(self, x: torch.Tensor):
#         assert x.dim() == 2 and x.size(1) == self.in_features

#         base_output = F.linear(self.base_activation(x), self.base_weight)  # (B,out)
#         spline_output = F.linear(
#             self.b_splines(x).view(x.size(0), -1),
#             self.scaled_spline_weight.view(self.out_features, -1),
#         )
#         return base_output + spline_output

#     @torch.no_grad()
#     def update_grid(self, x: torch.Tensor, margin=0.01):
#         assert x.dim() == 2 and x.size(1) == self.in_features
#         batch = x.size(0)

#         splines = self.b_splines(x)                # (B,in,coeff)
#         splines = splines.permute(1, 0, 2)         # (in,B,coeff)
#         orig_coeff = self.scaled_spline_weight     # (out,in,coeff)
#         orig_coeff = orig_coeff.permute(1, 2, 0)   # (in,coeff,out)
#         unreduced = torch.bmm(splines, orig_coeff) # (in,B,out)
#         unreduced = unreduced.permute(1, 0, 2)     # (B,in,out)

#         x_sorted = torch.sort(x, dim=0)[0]
#         grid_adaptive = x_sorted[
#             torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
#         ]

#         uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
#         grid_uniform = (
#             torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
#             * uniform_step
#             + x_sorted[0]
#             - margin
#         )

#         grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
#         grid = torch.cat(
#             [
#                 grid[:1]
#                 - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
#                 grid,
#                 grid[-1:]
#                 + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
#             ],
#             dim=0,
#         )

#         self.grid.copy_(grid.T)
#         self.spline_weight.data.copy_(self.curve2coeff(x, unreduced))

#     def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#         l1_fake = self.spline_weight.abs().mean(-1)  # (out,in)
#         reg_act = l1_fake.sum()
#         p = l1_fake / (reg_act + 1e-12)
#         reg_ent = -torch.sum(p * (p + 1e-12).log())
#         return regularize_activation * reg_act + regularize_entropy * reg_ent


# class KAN(nn.Module):
#     def __init__(
#         self,
#         layers_hidden,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         base_activation=nn.SiLU,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#     ):
#         """
#         layers_hidden: list like [in_dim, h1, h2, ..., out_dim]
#         """
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
#             self.layers.append(
#                 KANLinear(
#                     in_features=in_features,
#                     out_features=out_features,
#                     grid_size=grid_size,
#                     spline_order=spline_order,
#                     scale_noise=scale_noise,
#                     scale_base=scale_base,
#                     scale_spline=scale_spline,
#                     enable_standalone_scale_spline=True,
#                     base_activation=base_activation,
#                     grid_eps=grid_eps,
#                     grid_range=grid_range,
#                 )
#             )

#     def forward(self, x: torch.Tensor, update_grid=False):
#         for layer in self.layers:
#             if update_grid:
#                 layer.update_grid(x)
#             x = layer(x)
#         return x

#     def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#         return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)


# # =========================================================
# # 3) KAN fusion block: fuse (feature, message) per-sample (NO batch mixing)
# # =========================================================
# class KANFusionBlock(nn.Module):
#     """
#     per-sample fusion, NO batch mixing
#     f' = LN( f + alpha * KAN([f, msg]) )
#     """
#     def __init__(self,
#                  dim=512,
#                  hidden=1024,
#                  grid_size=5,
#                  spline_order=3,
#                  alpha=0.3,
#                  drop=0.1):
#         super().__init__()
#         self.alpha = alpha
#         self.drop = nn.Dropout(drop)

#         # Input is concat([f,msg]) => 2*dim
#         self.kan = KAN(
#             layers_hidden=[dim * 2, hidden, dim],
#             grid_size=grid_size,
#             spline_order=spline_order,
#         )
#         self.ln = nn.LayerNorm(dim)

#     def forward(self, f, msg, alpha=None, update_grid=False):
#         if alpha is None:
#             alpha = self.alpha
#         x = torch.cat([f, msg], dim=1)  # (B, 2D)
#         delta = self.kan(x, update_grid=update_grid)  # (B, D)
#         delta = self.drop(delta)
#         return self.ln(f + alpha * delta)


# # =========================================================
# # 4) Full module: BP bipartite MP -> KAN fusion (outputs separated img/txt)
# # =========================================================
# class BPThenKAN(nn.Module):
#     """
#     f_img, f_txt -> bipartite MP (get msg_i/msg_t) -> KAN fusion per modality -> (f_img', f_txt')
#     """
#     def __init__(self,
#                  dim=512,
#                  # bipartite
#                  tau=0.07,
#                  topk=16,
#                  drop_mp=0.0,
#                  use_proj=True,
#                  detach_affinity=True,
#                  alpha_mp=1.0,
#                  beta=0.1,
#                  use_bottom_k=False,
#                  # KAN
#                  hidden=1024,
#                  grid_size=5,
#                  spline_order=3,
#                  alpha_kan=0.3,
#                  drop_kan=0.1):
#         super().__init__()

#         self.mp = BPCompatibleBipartiteMP(
#             dim=dim,
#             tau=tau,
#             topk=topk,
#             drop=drop_mp,
#             use_proj=use_proj,
#             detach_affinity=detach_affinity,
#             alpha=alpha_mp,
#             beta=beta,
#             use_bottom_k=use_bottom_k
#         )

#         self.fuse_i = KANFusionBlock(
#             dim=dim,
#             hidden=hidden,
#             grid_size=grid_size,
#             spline_order=spline_order,
#             alpha=alpha_kan,
#             drop=drop_kan
#         )
#         self.fuse_t = KANFusionBlock(
#             dim=dim,
#             hidden=hidden,
#             grid_size=grid_size,
#             spline_order=spline_order,
#             alpha=alpha_kan,
#             drop=drop_kan
#         )

#     def forward(self, f_img, f_txt, update_grid=False,
#                 alpha_mp=None, alpha_kan=None):
#         # 1) cross-modal sparse bipartite message passing
#         f_img2, f_txt2, msg_i, msg_t = self.mp(
#             f_img, f_txt,
#             alpha=alpha_mp,
#             return_msg=True
#         )

#         # 2) KAN fusion per modality using the cross-modal aggregated message
#         f_img3 = self.fuse_i(f_img2, msg_i, alpha=alpha_kan, update_grid=update_grid)
#         f_txt3 = self.fuse_t(f_txt2, msg_t, alpha=alpha_kan, update_grid=update_grid)
#         return f_img3, f_txt3



