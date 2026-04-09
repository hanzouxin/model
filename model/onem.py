

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_softmax(x: torch.Tensor, dim: int):
    # fp16/bf16 下 softmax 用 float32 更稳
    return torch.softmax(x.float(), dim=dim).to(x.dtype)


def entropy(p: torch.Tensor, eps: float = 1e-12):
    return -(p * (p + eps).log()).sum(dim=-1)

class HGNNLayer(nn.Module):
    """
    X' = LN( X + alpha * Drop( Linear( Dv^-1/2 * H * De^-1 * H^T * Dv^-1/2 * X ) ) )
    X: (N,D), H: (N,E)
    """
    def __init__(self, dim=512, drop=0.1, alpha=1.0, act=True):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(drop)
        self.ln = nn.LayerNorm(dim)
        self.alpha = alpha
        self.act = act

    def forward(self, X: torch.Tensor, H: torch.Tensor, eps: float = 1e-6):
        # compute degrees in float32 for stability
        dv = H.sum(dim=1).float()  # (N,)
        de = H.sum(dim=0).float()  # (E,)
        dv_inv_sqrt = (dv + eps).rsqrt().to(X.dtype)
        de_inv = (de + eps).reciprocal().to(X.dtype)

        # Dv^-1/2 X
        X0 = X * dv_inv_sqrt.unsqueeze(1)      # (N,D)

        # H^T X0  then De^-1
        Xe = H.transpose(0, 1) @ X0            # (E,D)
        Xe = Xe * de_inv.unsqueeze(1)          # (E,D)

        # H Xe then Dv^-1/2
        X1 = H @ Xe                            # (N,D)
        X1 = X1 * dv_inv_sqrt.unsqueeze(1)     # (N,D)

        out = self.fc(X1)
        if self.act:
            out = F.gelu(out)
        out = self.drop(out)

        return self.ln(X + self.alpha * out)


# =========================================================
# 2) Build cross-modal hypergraph incidence H
# =========================================================
def build_cross_modal_hypergraph(
    f_img: torch.Tensor,
    f_txt: torch.Tensor,
    topk: int = 8,
    tau: float = 0.07,
    self_loop_weight: float = 2.0,
    soft: bool = True,
    mutual: bool = True,
    min_sim: float = None,
):
    """
    Nodes: N = Bi + Bt
      0..Bi-1         : images
      Bi..Bi+Bt-1     : texts

    Hyperedges: E = Bi + Bt
      e=i      : img_i + topk texts
      e=Bi+j   : txt_j + topk images

    Return:
      H:(N,E), S:(Bi,Bt), idx_t:(Bi,k), idx_i:(Bt,k), w_t:(Bi,k), w_i:(Bt,k)
    """
    device = f_img.device
    Bi, D = f_img.shape
    Bt, _ = f_txt.shape
    N = Bi + Bt
    E = Bi + Bt

    # cosine sim
    fi = F.normalize(f_img, dim=-1)
    ft = F.normalize(f_txt, dim=-1)
    S = (fi @ ft.t()) / tau  # (Bi,Bt)

    k_t = min(topk, Bt)
    k_i = min(topk, Bi)

    idx_t = S.topk(k_t, dim=1).indices                      # (Bi,k_t)
    idx_i = S.topk(k_i, dim=0).indices.t().contiguous()     # (Bt,k_i)

    # mutual top-k filtering
    if mutual:
        A = torch.zeros((Bi, Bt), device=device, dtype=torch.bool)
        A.scatter_(1, idx_t, True)
        B = torch.zeros((Bt, Bi), device=device, dtype=torch.bool)
        B.scatter_(1, idx_i, True)
        M = A & B.t()
        if min_sim is not None:
            M = M & (S > min_sim)

        neg_large = -1e4 if S.dtype in (torch.float16, torch.bfloat16) else -1e9
        S_m = S.masked_fill(~M, neg_large)
        idx_t = S_m.topk(k_t, dim=1).indices
        idx_i = S_m.topk(k_i, dim=0).indices.t().contiguous()

    if soft:
        w_t = safe_softmax(S.gather(1, idx_t), dim=1)         # (Bi,k_t)
        w_i = safe_softmax(S.t().gather(1, idx_i), dim=1)     # (Bt,k_i)
    else:
        w_t = torch.full((Bi, k_t), 1.0 / max(k_t, 1), device=device, dtype=f_img.dtype)
        w_i = torch.full((Bt, k_i), 1.0 / max(k_i, 1), device=device, dtype=f_img.dtype)

    # incidence H (N,E)
    H = torch.zeros((N, E), device=device, dtype=f_img.dtype)

    # image-centered edges
    for i in range(Bi):
        e = i
        H[i, e] = self_loop_weight
        js = idx_t[i]
        H[Bi + js, e] = w_t[i]

    # text-centered edges
    for j in range(Bt):
        e = Bi + j
        H[Bi + j, e] = self_loop_weight
        is_ = idx_i[j]
        H[is_, e] = w_i[j]

    return H, S, idx_t, idx_i, w_t, w_i


# =========================================================
# 3) Cross-Modal HGNN Only 
# =========================================================
class CrossModalHGNN(nn.Module):
    """
    Hypergraph propagation only.
    Input : f_img(B,D), f_txt(B,D)
    Output: f_img'(B,D), f_txt'(B,D)
    """
    def __init__(self,
                 dim=512,
                 topk=8,
                 tau=0.07,
                 self_loop_weight=2.0,
                 soft=True,
                 mutual=True,
                 min_sim=None,
                 layers=1,
                 drop=0.1,
                 alpha=1.0,
                 detach_H=True):
        super().__init__()
        self.dim = dim
        self.topk = topk
        self.tau = tau
        self.self_loop_weight = self_loop_weight
        self.soft = soft
        self.mutual = mutual
        self.min_sim = min_sim
        self.detach_H = detach_H

        self.layers = nn.ModuleList([HGNNLayer(dim, drop, alpha, act=True) for _ in range(layers)])
        self.ln_img = nn.LayerNorm(dim)
        self.ln_txt = nn.LayerNorm(dim)

    def forward(self, f_img: torch.Tensor, f_txt: torch.Tensor):
        Bi = f_img.size(0)
        Bt = f_txt.size(0)

        H, S, idx_t, idx_i, w_t, w_i = build_cross_modal_hypergraph(
            f_img, f_txt,
            topk=self.topk,
            tau=self.tau,
            self_loop_weight=self.self_loop_weight,
            soft=self.soft,
            mutual=self.mutual,
            min_sim=self.min_sim,
        )
        if self.detach_H:
            H = H.detach()

        X = torch.cat([f_img, f_txt], dim=0)  # (N,D)
        for layer in self.layers:
            X = layer(X, H)

        out_img = self.ln_img(X[:Bi])
        out_txt = self.ln_txt(X[Bi:Bi+Bt])
        return out_img, out_txt

# # =========================================================
# # 4) Explicit Cross Interaction (img<->txt) via Top-k attention (manual)


# class LowRankBilinearFusion(nn.Module):

#     def __init__(self, dim=512, rank=1024, drop=0.1, alpha=1.0, use_ln=True):
#         super().__init__()
#         self.alpha = alpha
#         self.drop = nn.Dropout(drop)

#         self.pi = nn.Linear(dim, rank, bias=False)
#         self.pt = nn.Linear(dim, rank, bias=False)

#         self.oi = nn.Linear(rank, dim, bias=False)
#         self.ot = nn.Linear(rank, dim, bias=False)

#         self.ln_i = nn.LayerNorm(dim) if use_ln else nn.Identity()
#         self.ln_t = nn.LayerNorm(dim) if use_ln else nn.Identity()

#     def forward(self, f_img, f_txt, alpha=None):
#         if alpha is None:
#             alpha = self.alpha

#         zi = self.pi(f_img)          # (B,rank)
#         zt = self.pt(f_txt)          # (B,rank)
#         z  = zi * zt                 # ✅ 高阶交互 (B,rank)
#         z  = self.drop(F.gelu(z))

#         delta_i = self.oi(z)         # (B,dim)
#         delta_t = self.ot(z)

#         out_i = self.ln_i(f_img + alpha * delta_i)
#         out_t = self.ln_t(f_txt + alpha * delta_t)
#         return out_i, out_t

# # =========================================================
# # 5) Full: HGNN -> Cross Interaction
# # =========================================================
# class HGNNThenCrossInteract(nn.Module):
#     """
#     Full module you want:
#       HGNN (hypergraph) -> cross interaction (img<->txt)
#     """
#     def __init__(self,
#                  dim=512,
#                  # HGNN
#                  h_topk=8,
#                  h_tau=0.07,
#                  h_self_loop_weight=2.0,
#                  h_soft=True,
#                  h_mutual=True,
#                  h_min_sim=None,
#                  h_layers=1,
#                  h_drop=0.1,
#                  h_alpha=1.0,
#                  h_detach_H=True,
#                  # Interaction
#                  ):
#         super().__init__()
#         self.hgnn = CrossModalHGNNOnly(
#             dim=dim,
#             topk=h_topk,
#             tau=h_tau,
#             self_loop_weight=h_self_loop_weight,
#             soft=h_soft,
#             mutual=h_mutual,
#             min_sim=h_min_sim,
#             layers=h_layers,
#             drop=h_drop,
#             alpha=h_alpha,
#             detach_H=h_detach_H
#         )
#         self.inter = LowRankBilinearFusion(dim=dim, rank=1024, alpha=1.0)

#     def forward(self, f_img: torch.Tensor, f_txt: torch.Tensor):
#         f_img_h, f_txt_h = self.hgnn(f_img, f_txt)
#         f_img_o, f_txt_o = self.inter(f_img_h, f_txt_h)

#         # with torch.no_grad():
#         #     delta_img_ratio = ((f_img_o - f_img_h).norm(dim=1) / (f_img_h.norm(dim=1) + 1e-6)).mean().item()
#         #     delta_txt_ratio = ((f_txt_o - f_txt_h).norm(dim=1) / (f_txt_h.norm(dim=1) + 1e-6)).mean().item()
#         #     print(f"[dbg] delta_img_ratio={delta_img_ratio:.4f}  delta_txt_ratio={delta_txt_ratio:.4f}")
#         return f_img_o, f_txt_o



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
from torch.nn.functional import normalize

# class DiffusionLayer(nn.Module):
#     def __init__(self, step_size):
#         super(DiffusionLayer, self).__init__()
#         self.step_size = step_size

#     def forward(self, x, laplacian):
#         # 扩散过程: x - step_size * Laplacian * x
#         x = x - self.step_size * torch.matmul(laplacian, x.flatten(1)).view_as(x)
#         return x


# class JointDiffusionResNet(nn.Module):
#     def __init__(self, img_dim=512, txt_dim=512, step_size=0.1, layer_num=3,mlp_hidden_dim=4096):
#         super(JointDiffusionResNet, self).__init__()

#         # 扩散层
#         self.diffusion_layer = DiffusionLayer(step_size)

#         # 图像和文本的初始特征映射（映射到同一维度）
#         # self.fc_img = nn.Linear(img_dim, img_dim)
#         # self.fc_txt = nn.Linear(txt_dim, txt_dim)

#         # 残差连接层
#         self.fc_residual_img = nn.Linear(img_dim, img_dim)
#         self.fc_residual_txt = nn.Linear(txt_dim, txt_dim)
#         # self.lambda_d = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)
#         self.lambda_d = 0.1
#         #归一化层
#         self.bn_img = nn.BatchNorm1d(img_dim)
#         self.bn_txt = nn.BatchNorm1d(txt_dim)
        

#         self.layer_num = layer_num
    

#     def knn_graph(self, features, k=5):
#         """
#         使用 K-近邻算法生成基于样本特征的图的权重矩阵。
#         :param features: 输入特征，形状为 (num_samples, feature_dim)
#         :param k: 近邻数量
#         :return: 权重矩阵，形状为 (num_samples, num_samples)
#         """
#         knn = NearestNeighbors(n_neighbors=k, metric='cosine')
#         knn.fit(features)
        
#         distances, indices = knn.kneighbors(features)

#         num_samples = features.shape[0]
#         weight_matrix = np.zeros((num_samples, num_samples))

#         for i in range(num_samples):
#             for j in range(1, k):  # 忽略自己本身
#                 weight_matrix[i, indices[i][j]] = np.exp(-distances[i][j])  # 使用高斯核计算权重

#         return torch.tensor(weight_matrix, dtype=torch.float32)

#     def forward(self, img_feat, txt_feat):
        
#         # 初始化联合特征 (concatenate 图像和文本特征)
#         joint_feat = torch.cat([img_feat, txt_feat], dim=0)

#         # 动态计算图像和文本的 KNN 图的拉普拉斯矩阵
#         weight_img = self.knn_graph(img_feat.detach().cpu().numpy())
#         weight_txt = self.knn_graph(txt_feat.detach().cpu().numpy())

#         diagonal_img = torch.diag(weight_img.sum(dim=1))
#         laplacian_img = diagonal_img - weight_img

#         diagonal_txt = torch.diag(weight_txt.sum(dim=1))
#         laplacian_txt = diagonal_txt - weight_txt

#         # 构建联合图的拉普拉斯矩阵（图像和文本）
#         laplacian_joint = torch.block_diag(laplacian_img, laplacian_txt).to(joint_feat.device)

#         # 残差网络和扩散过程
#         for _ in range(self.layer_num):
#             joint_feat_res = F.relu(self.fc_residual_img(joint_feat))  # 残差连接
#             joint_feat = joint_feat + joint_feat_res  # 残差输出
#             joint_feat = self.diffusion_layer(joint_feat, laplacian_joint)  # 扩散层

#         # 分离图像和文本的特征
#         img_feat = self.bn_img(joint_feat[:img_feat.shape[0], :])*self.lambda_d + img_feat
#         txt_feat = self.bn_txt(joint_feat[img_feat.shape[0]:, :])*self.lambda_d + txt_feat

#         return img_feat, txt_feat
    


    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class KANEncoder(nn.Module):
#     def __init__(self, dim=512):
#         super().__init__()
#         # ✅ token 是 512 维，所以 KAN 的 in_features 也必须是 512
#         self.dim = dim
#         self.kan = KAN([dim, dim])  # 你也可以用 [dim, dim, dim]

#     def feature_enhance(self, image_embed, text_embed):
#         """
#         image_embed: [B,N,D]
#         text_embed:  [B,L,D]
#         输出: feature_c: [B, N+L, D]
#         """
#         # 用全局（从局部求和/平均）构造 batch 相似矩阵
#         # （你原来 sum(dim=1) 没问题，但建议 mean 更稳定）
#         i1 = torch.mean(image_embed, dim=1)  # [B,D]
#         t1 = torch.mean(text_embed,  dim=1)  # [B,D]

#         mi = i1 @ i1.t()  # [B,B]
#         mt = t1 @ t1.t()  # [B,B]
#         similar_matrix = mi - mt  # [B,B]

#         # 你的稳定化非线性
#         similar_matrix = (1 - torch.tanh(similar_matrix) ** 2) * \
#                          torch.sigmoid(similar_matrix) * (1 - torch.sigmoid(similar_matrix))

#         # ✅ 用 [B,B] 去“跨样本重加权” token：得到 [B,N,D] / [B,L,D]
#         feature_a = torch.einsum('ij,jnd->ind', similar_matrix, image_embed)  # [B,N,D]
#         feature_b = torch.einsum('ij,jld->ild', similar_matrix, text_embed)   # [B,L,D]

#         # 拼接 token 维度
#         feature_c = torch.cat((feature_a, feature_b), dim=1)  # [B,N+L,D]
#         return feature_c

#     def forward(self, image_embed, text_embed):
#         """
#         image_embed: [B,N,D]
#         text_embed:  [B,L,D]
#         return: refined_image_tokens [B,N,D], refined_text_tokens [B,L,D]
#         """
#         B, N, D = image_embed.shape
#         _, L, _ = text_embed.shape
#         assert D == self.dim, f"token dim mismatch: got {D}, expected {self.dim}"

#         # 1) 拼成一个 token 序列
#         tokens = torch.cat((image_embed, text_embed), dim=1)  # [B,N+L,D]

#         # 2) ✅ flatten 成 2D，喂给 KAN
#         x = tokens.reshape(B * (N + L), D)         # [B*(N+L), D]
#         y = self.kan(x)                             # [B*(N+L), D]
#         result = y.reshape(B, (N + L), D)           # [B,N+L,D]

#         # 3) 论文里的 + [S'Fv, S'Ft]
#         result = result + self.feature_enhance(image_embed, text_embed)  # [B,N+L,D]

#         # ✅ token 归一化要在最后一维 dim=-1（不是 dim=1）
#         result = F.normalize(result, p=2, dim=-1)

#         # 4) 切回图像 token / 文本 token
#         out_img = result[:, :N, :]   # [B,N,D]
#         out_txt = result[:, N:, :]   # [B,L,D]
#         return out_img, out_txt

    


# class KANLinear(torch.nn.Module):
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
#         base_activation=torch.nn.SiLU,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#     ):
#         super(KANLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.grid_size = grid_size
#         self.spline_order = spline_order

#         h = (grid_range[1] - grid_range[0]) / grid_size
#         grid = (
#             (
#                 torch.arange(-spline_order, grid_size + spline_order + 1) * h
#                 + grid_range[0]
#             )
#             .expand(in_features, -1)
#             .contiguous()
#         )
#         self.register_buffer("grid", grid)

#         self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
#         self.spline_weight = torch.nn.Parameter(
#             torch.Tensor(out_features, in_features, grid_size + spline_order)
#         )
#         if enable_standalone_scale_spline:
#             self.spline_scaler = torch.nn.Parameter(
#                 torch.Tensor(out_features, in_features)
#             )

#         self.scale_noise = scale_noise
#         self.scale_base = scale_base
#         self.scale_spline = scale_spline
#         self.enable_standalone_scale_spline = enable_standalone_scale_spline
#         self.base_activation = base_activation()
#         self.grid_eps = grid_eps

#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
#         with torch.no_grad():
#             noise = (
#                 (
#                     torch.rand(self.grid_size + 1, self.in_features, self.out_features)
#                     - 1 / 2
#                 )
#                 * self.scale_noise
#                 / self.grid_size
#             )
#             self.spline_weight.data.copy_(
#                 (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
#                 * self.curve2coeff(
#                     self.grid.T[self.spline_order : -self.spline_order],
#                     noise,
#                 )
#             )
#             if self.enable_standalone_scale_spline:
#                 # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
#                 torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

#     def b_splines(self, x: torch.Tensor):
#         """
#         Compute the B-spline bases for the given input tensor.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, in_features).

#         Returns:
#             torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
#         """
#         assert x.dim() == 2 and x.size(1) == self.in_features

#         grid: torch.Tensor = (
#             self.grid
#         )  # (in_features, grid_size + 2 * spline_order + 1)
#         x = x.unsqueeze(-1)
#         bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
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

#         assert bases.size() == (
#             x.size(0),
#             self.in_features,
#             self.grid_size + self.spline_order,
#         )
#         return bases.contiguous()

#     def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
#         """
#         Compute the coefficients of the curve that interpolates the given points.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, in_features).
#             y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

#         Returns:
#             torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
#         """
#         assert x.dim() == 2 and x.size(1) == self.in_features
#         assert y.size() == (x.size(0), self.in_features, self.out_features)

#         A = self.b_splines(x).transpose(
#             0, 1
#         )  # (in_features, batch_size, grid_size + spline_order)
#         B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
#         solution = torch.linalg.lstsq(
#             A, B
#         ).solution  # (in_features, grid_size + spline_order, out_features)
#         result = solution.permute(
#             2, 0, 1
#         )  # (out_features, in_features, grid_size + spline_order)

#         assert result.size() == (
#             self.out_features,
#             self.in_features,
#             self.grid_size + self.spline_order,
#         )
#         return result.contiguous()

#     @property
#     def scaled_spline_weight(self):
#         return self.spline_weight * (
#             self.spline_scaler.unsqueeze(-1)
#             if self.enable_standalone_scale_spline
#             else 1.0
#         )

#     def forward(self, x: torch.Tensor):
#         assert x.dim() == 2 and x.size(1) == self.in_features
#         base_output = F.linear(self.base_activation(x), self.base_weight)
#         spline_output = F.linear(
#             self.b_splines(x).view(x.size(0), -1),
#             self.scaled_spline_weight.view(self.out_features, -1),
#         )
#         return base_output + spline_output

#     @torch.no_grad()
#     def update_grid(self, x: torch.Tensor, margin=0.01):
#         assert x.dim() == 2 and x.size(1) == self.in_features
#         batch = x.size(0)

#         splines = self.b_splines(x)  # (batch, in, coeff)
#         splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
#         orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
#         orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
#         unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
#         unreduced_spline_output = unreduced_spline_output.permute(
#             1, 0, 2
#         )  # (batch, in, out)

#         # sort each channel individually to collect data distribution
#         x_sorted = torch.sort(x, dim=0)[0]
#         grid_adaptive = x_sorted[
#             torch.linspace(
#                 0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
#             )
#         ]

#         uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
#         grid_uniform = (
#             torch.arange(
#                 self.grid_size + 1, dtype=torch.float32, device=x.device
#             ).unsqueeze(1)
#             * uniform_step
#             + x_sorted[0]
#             - margin
#         )

#         grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
#         grid = torch.concatenate(
#             [
#                 grid[:1]
#                 - uniform_step
#                 * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
#                 grid,
#                 grid[-1:]
#                 + uniform_step
#                 * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
#             ],
#             dim=0,
#         )

#         self.grid.copy_(grid.T)
#         self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

#     def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
#         """
#         Compute the regularization loss.

#         This is a dumb simulation of the original L1 regularization as stated in the
#         paper, since the original one requires computing absolutes and entropy from the
#         expanded (batch, in_features, out_features) intermediate tensor, which is hidden
#         behind the F.linear function if we want an memory efficient implementation.

#         The L1 regularization is now computed as mean absolute value of the spline
#         weights. The authors implementation also includes this term in addition to the
#         sample-based regularization.
#         """
#         l1_fake = self.spline_weight.abs().mean(-1)
#         regularization_loss_activation = l1_fake.sum()
#         p = l1_fake / regularization_loss_activation
#         regularization_loss_entropy = -torch.sum(p * p.log())
#         return (
#             regularize_activation * regularization_loss_activation
#             + regularize_entropy * regularization_loss_entropy
#         )


# class KAN(torch.nn.Module):
#     def __init__(
#         self,
#         layers_hidden,
#         grid_size=5,
#         spline_order=3,
#         scale_noise=0.1,
#         scale_base=1.0,
#         scale_spline=1.0,
#         base_activation=torch.nn.SiLU,
#         grid_eps=0.02,
#         grid_range=[-1, 1],
#     ):
#         super(KAN, self).__init__()
#         self.grid_size = grid_size
#         self.spline_order = spline_order

#         self.layers = torch.nn.ModuleList()
        
#         for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
#             self.layers.append(
#                 KANLinear(
#                     in_features,
#                     out_features,
#                     grid_size=grid_size,
#                     spline_order=spline_order,
#                     scale_noise=scale_noise,
#                     scale_base=scale_base,
#                     scale_spline=scale_spline,
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
#         return sum(
#             layer.regularization_loss(regularize_activation, regularize_entropy)
#             for layer in self.layers
#         )

