# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class MSLoss(nn.Module):
#     def __init__(self, temperature=0.3, totalepoch=100, self_paced=True):
#         super(MSLoss, self).__init__()
#         self.temperature = temperature
#         self.totalepoch = totalepoch
#         self.self_paced = self_paced

#     def forward(self, image_feature, text_feature, labels=None, epoch=0):

#         mask = (torch.mm(labels.float(), labels.float().T) > 0).float()
#         pos_mask = mask
#         neg_mask = 1 - mask

#         image_dot_text = torch.matmul(F.normalize(image_feature, dim=1), F.normalize(text_feature, dim=1).T)

#         all_exp = torch.exp(image_dot_text / self.temperature)
#         pos_exp = pos_mask * all_exp
#         neg_exp = neg_mask * all_exp

#         if self.self_paced:
#             if epoch <= int(self.totalepoch/3):
#                 delta = epoch / int(self.totalepoch/3)
#             else:
#                 delta = 1
#             pos_exp *= torch.exp(-1 - image_dot_text).detach() ** (delta/4)
#             neg_exp *= torch.exp(-1 + image_dot_text).detach() ** (delta)

#         loss = -torch.log(pos_exp.sum(1)/(neg_exp.sum(1) + pos_exp.sum(1)))
#         return loss.mean()


# import torch
# import torch.nn as nn

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class MSGuidedSelfPacedInfoNCE(nn.Module):
#     """
#     v2: stable + stronger
#     - neg weights: stop-grad (stable)
#     - pos weights: allow gradient (recover performance)
#     - log-weight clamp: self-paced (cap_min -> cap_max)
#     """
#     def __init__(self,
#                  temperature=0.2,
#                  totalepoch=100,
#                  ramp_ratio=1/3,
#                  thresh=0.0,
#                  scale_pos=2.0,
#                  scale_neg=40.0,
#                  clamp_exp=50.0,
#                  cap_min=2.0,
#                  cap_max=6.0):
#         super().__init__()
#         self.tau = temperature
#         self.totalepoch = totalepoch
#         self.ramp_ratio = ramp_ratio

#         self.thresh = thresh
#         self.scale_pos = scale_pos
#         self.scale_neg = scale_neg

#         self.clamp_exp = clamp_exp
#         self.cap_min = cap_min
#         self.cap_max = cap_max

#     def _delta(self, epoch: int):
#         ramp = max(1, int(self.totalepoch * self.ramp_ratio))
#         return min(1.0, float(epoch) / float(ramp))

#     def _masked_mean(self, x, mask, dim=1, eps=1e-12):
#         mask_f = mask.float()
#         s = (x * mask_f).sum(dim=dim)
#         c = mask_f.sum(dim=dim).clamp_min(1.0)
#         return s / (c + eps)

#     def _one_direction(self, sim, pos_mask, neg_mask, delta):
#         # logits
#         logits = sim / self.tau

#         # MS-style difficulty scores
#         dp = torch.exp(torch.clamp(-self.scale_pos * (sim - self.thresh), max=self.clamp_exp))
#         dn = torch.exp(torch.clamp( self.scale_neg * (sim - self.thresh), max=self.clamp_exp))

#         # ✅ 关键：neg 权重 stop-grad（更稳）
#         dn_w = dn.detach()
#         # ✅ 关键：pos 权重允许梯度（更强）
#         dp_w = dp  # keep grad

#         # 统计均值：仍然 detach（避免均值统计引入不稳定梯度）
#         mp = self._masked_mean(dp.detach(), pos_mask, dim=1).unsqueeze(1)
#         mn = self._masked_mean(dn_w,        neg_mask, dim=1).unsqueeze(1)

#         wp = (dp_w / (mp + 1e-12)).clamp_min(1e-12)
#         wn = (dn_w / (mn + 1e-12)).clamp_min(1e-12)

#         # ✅ self-paced clamp：前期保守、后期放开一点（拉回效果）
#         cap = self.cap_min + (self.cap_max - self.cap_min) * delta
#         log_wp = torch.clamp(delta * torch.log(wp), -cap, cap)
#         log_wn = torch.clamp(delta * torch.log(wn), -cap, cap)

#         neg_inf = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)

#         # numerator: logsumexp over positives
#         logw_pos_only = torch.where(pos_mask, log_wp, neg_inf)
#         log_num = torch.logsumexp(logits + logw_pos_only, dim=1)

#         # denominator: positives use log_wp, negatives use log_wn
#         logw_all = torch.where(pos_mask, log_wp, torch.where(neg_mask, log_wn, neg_inf))
#         log_den = torch.logsumexp(logits + logw_all, dim=1)

#         valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
#         loss = -(log_num - log_den)
#         return loss[valid].mean() if valid.any() else sim.new_zeros(())

#     def forward(self, h1, h2, labels, dataset="MSLOSS", epoch=0):
#         # mask
#         if dataset == "cifar10-1":
#             y = torch.argmax(labels, dim=1)
#             pos_mask = (y[:, None] == y[None, :])
#         else:
#             lab = labels.float()
#             pos_mask = (lab @ lab.t()) > 0
#         neg_mask = ~pos_mask

#         # hash similarity: dot/B
#         B = h1.size(1)
#         sim = (h1 @ h2.t()) / float(B)

#         delta = self._delta(epoch)
#         i2t = self._one_direction(sim,   pos_mask,   neg_mask,   delta)
#         t2i = self._one_direction(sim.t(), pos_mask.t(), neg_mask.t(), delta)
#         return 0.5 * (i2t + t2i)




# import torch
# import torch.nn as nn

# class MSMinedGuidedSelfPacedInfoNCE(nn.Module):
#     """
#     V2->V3 direction: Keep MS-guided weighting, but ADD MS-style mining masks
#     so aux focuses on informative pairs (near the margin boundary).

#     - MS-mining (vectorized):
#         neg_keep: sim + margin > min_pos
#         pos_keep: sim - margin < max_neg
#     - neg weights stop-grad, pos weights allow grad
#     - self-paced clamp (cap_min->cap_max)
#     - optional: EMA thresh to avoid late-stage degeneration (recommended)
#     """
#     def __init__(self,
#                  temperature=0.2,
#                  totalepoch=100,
#                  ramp_ratio=1/3,
#                  thresh=0.0,
#                  use_ema_thresh=True,
#                  thresh_ema_m=0.9,
#                  scale_pos=2.0,
#                  scale_neg=40.0,
#                  clamp_exp=50.0,
#                  cap_min=2.0,
#                  cap_max=6.0,
#                  margin=0.1,
#                  delta_on_pos=False):
#         super().__init__()
#         self.tau = temperature
#         self.totalepoch = totalepoch
#         self.ramp_ratio = ramp_ratio

#         self.thresh_init = float(thresh)
#         self.use_ema_thresh = use_ema_thresh
#         self.thresh_ema_m = thresh_ema_m
#         self.register_buffer("thresh_ema", torch.tensor(self.thresh_init))

#         self.scale_pos = scale_pos
#         self.scale_neg = scale_neg
#         self.clamp_exp = clamp_exp

#         self.cap_min = cap_min
#         self.cap_max = cap_max

#         self.margin = margin
#         self.delta_on_pos = delta_on_pos

#     def _delta(self, epoch: int):
#         ramp = max(1, int(self.totalepoch * self.ramp_ratio))
#         return min(1.0, float(epoch) / float(ramp))

#     def _masked_mean(self, x, mask, dim=1, eps=1e-12):
#         mask_f = mask.float()
#         s = (x * mask_f).sum(dim=dim)
#         c = mask_f.sum(dim=dim).clamp_min(1.0)
#         return s / (c + eps)

#     @torch.no_grad()
#     def _update_thresh_ema(self, sim, pos_mask, neg_mask):
#         pos_vals = sim[pos_mask]
#         neg_vals = sim[neg_mask]
#         if pos_vals.numel() == 0 or neg_vals.numel() == 0:
#             return
#         t_batch = 0.5 * (pos_vals.mean() + neg_vals.mean())
#         self.thresh_ema.mul_(self.thresh_ema_m).add_((1 - self.thresh_ema_m) * t_batch)

#     def _ms_mining_masks(self, sim, pos_mask, neg_mask):
#         # min_pos per row
#         pos_sim = sim.masked_fill(~pos_mask, float("inf"))
#         min_pos = pos_sim.min(dim=1).values

#         # max_neg per row
#         neg_sim = sim.masked_fill(~neg_mask, float("-inf"))
#         max_neg = neg_sim.max(dim=1).values

#         # anchors that have both
#         valid = torch.isfinite(min_pos) & torch.isfinite(max_neg)

#         # MS-style keep masks (vectorized)
#         neg_keep = neg_mask & (sim + self.margin > min_pos.unsqueeze(1))
#         pos_keep = pos_mask & (sim - self.margin < max_neg.unsqueeze(1))

#         return pos_keep, neg_keep, valid

#     def _one_direction(self, sim, pos_mask, neg_mask, delta, thresh):
#         logits = sim / self.tau

#         # ✅ mining: focus on informative pairs
#         pos_keep, neg_keep, valid_anchor = self._ms_mining_masks(sim, pos_mask, neg_mask)
#         pos_keep = pos_keep & valid_anchor.unsqueeze(1)
#         neg_keep = neg_keep & valid_anchor.unsqueeze(1)

#         # MS-guided difficulty scores (only computed once, masked later)
#         dp = torch.exp(torch.clamp(-self.scale_pos * (sim - thresh), max=self.clamp_exp))
#         dn = torch.exp(torch.clamp( self.scale_neg * (sim - thresh), max=self.clamp_exp))

#         dn_w = dn.detach()
#         dp_w = dp  # keep grad

#         # normalize weights over "kept" pairs (important!)
#         mp = self._masked_mean(dp.detach(), pos_keep, dim=1).unsqueeze(1)
#         mn = self._masked_mean(dn_w,        neg_keep, dim=1).unsqueeze(1)

#         wp = (dp_w / (mp + 1e-12)).clamp_min(1e-12)
#         wn = (dn_w / (mn + 1e-12)).clamp_min(1e-12)

#         # self-paced clamp
#         cap = self.cap_min + (self.cap_max - self.cap_min) * delta
#         d_pos = delta if self.delta_on_pos else 1.0
#         d_neg = delta

#         log_wp = torch.clamp(d_pos * torch.log(wp), -cap, cap)
#         log_wn = torch.clamp(d_neg * torch.log(wn), -cap, cap)

#         neg_inf = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)

#         # numerator: only kept positives
#         logw_pos_only = torch.where(pos_keep, log_wp, neg_inf)
#         log_num = torch.logsumexp(logits + logw_pos_only, dim=1)

#         # denominator: kept positives + kept negatives
#         keep_all = pos_keep | neg_keep
#         logw_all = torch.where(pos_keep, log_wp, torch.where(neg_keep, log_wn, neg_inf))
#         log_den = torch.logsumexp(torch.where(keep_all, logits + logw_all, neg_inf), dim=1)

#         # valid: must have kept pos and kept neg
#         valid = (pos_keep.sum(dim=1) > 0) & (neg_keep.sum(dim=1) > 0)
#         loss = -(log_num - log_den)
#         return loss[valid].mean() if valid.any() else sim.new_zeros(())

#     def forward(self, h1, h2, labels, dataset="MSLOSS", epoch=0):
#         # mask
#         if dataset == "cifar10-1":
#             y = torch.argmax(labels, dim=1)
#             pos_mask = (y[:, None] == y[None, :])
#         else:
#             lab = labels.float()
#             pos_mask = (lab @ lab.t()) > 0
#         neg_mask = ~pos_mask

#         # hash similarity: dot/B
#         B = h1.size(1)
#         sim = (h1 @ h2.t()) / float(B)

#         delta = self._delta(epoch)

#         # ✅ adaptive thresh (recommended)
#         if self.use_ema_thresh and self.training:
#             self._update_thresh_ema(sim.detach(), pos_mask, neg_mask)
#             thresh = float(self.thresh_ema.item())
#         else:
#             thresh = self.thresh_init

#         i2t = self._one_direction(sim,      pos_mask,      neg_mask,      delta, thresh)
#         t2i = self._one_direction(sim.t(),  pos_mask.t(),  neg_mask.t(),  delta, thresh)
#         return 0.5 * (i2t + t2i)



# import torch
# import torch.nn as nn

# import torch
# import torch.nn as nn

# class MSMinedGuidedSelfPacedInfoNCE(nn.Module):
#     """
#     Robust V3 (keep your direction, fix large-dataset pain points):
#     - replace min_pos/max_neg with softmin/softmax (row-wise, masked)
#     - fallback if mined set empty (avoid starving)
#     - scale_neg curriculum (start->end) to reduce false-negative damage early
#     - keep: neg weight stop-grad, pos weight allow grad, self-paced clamp, EMA thresh
#     """
#     def __init__(self,
#                  temperature=0.2,
#                  totalepoch=100,
#                  ramp_ratio=1/3,

#                  thresh=0.0,
#                  use_ema_thresh=True,
#                  thresh_ema_m=0.9,

#                  scale_pos=2.0,
#                  scale_neg_end=40.0,        # ✅ 末期强度
#                  scale_neg_start=20.0,      # ✅ 早期更温和（大数据集更友好）
#                  clamp_exp=50.0,

#                  cap_min=2.0,
#                  cap_max=6.0,

#                  margin=0.1,
#                  delta_on_pos=False,

#                  # ✅ soft boundary sharpness (bigger -> closer to hard min/max)
#                  alpha_pos=10.0,
#                  alpha_neg=10.0):
#         super().__init__()
#         self.tau = temperature
#         self.totalepoch = totalepoch
#         self.ramp_ratio = ramp_ratio

#         self.thresh_init = float(thresh)
#         self.use_ema_thresh = use_ema_thresh
#         self.thresh_ema_m = thresh_ema_m
#         self.register_buffer("thresh_ema", torch.tensor(self.thresh_init))

#         self.scale_pos = scale_pos
#         self.scale_neg_start = scale_neg_start
#         self.scale_neg_end = scale_neg_end

#         self.clamp_exp = clamp_exp
#         self.cap_min = cap_min
#         self.cap_max = cap_max

#         self.margin = margin
#         self.delta_on_pos = delta_on_pos

#         self.alpha_pos = alpha_pos
#         self.alpha_neg = alpha_neg

#     def _delta(self, epoch: int):
#         ramp = max(1, int(self.totalepoch * self.ramp_ratio))
#         return min(1.0, float(epoch) / float(ramp))

#     def _masked_mean(self, x, mask, dim=1, eps=1e-12):
#         mask_f = mask.float()
#         s = (x * mask_f).sum(dim=dim)
#         c = mask_f.sum(dim=dim).clamp_min(1.0)
#         return s / (c + eps)

#     @torch.no_grad()
#     def _update_thresh_ema(self, sim, pos_mask, neg_mask):
#         # 更建议用边界统计更新阈值（比 mean(pos/neg) 更稳定）
#         min_pos, max_neg, valid = self._soft_boundary(sim, pos_mask, neg_mask)
#         if not valid.any():
#             return
#         t_batch = 0.5 * (min_pos[valid].mean() + max_neg[valid].mean())
#         self.thresh_ema.mul_(self.thresh_ema_m).add_((1 - self.thresh_ema_m) * t_batch)

#     def _soft_boundary(self, sim, pos_mask, neg_mask):
#         """
#         softmin over positives, softmax over negatives (row-wise)
#         returns: min_pos, max_neg, valid
#         """
#         neg_inf = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)

#         # softmin(pos): -1/a * logsumexp(-a*sim) over positives
#         z_pos = torch.where(pos_mask, -self.alpha_pos * sim, neg_inf)
#         lse_pos = torch.logsumexp(z_pos, dim=1)
#         min_pos = -lse_pos / self.alpha_pos

#         # softmax(neg):  1/a * logsumexp( a*sim) over negatives
#         z_neg = torch.where(neg_mask, self.alpha_neg * sim, neg_inf)
#         lse_neg = torch.logsumexp(z_neg, dim=1)
#         max_neg = lse_neg / self.alpha_neg

#         valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
#         return min_pos, max_neg, valid

#     def _ms_mining_masks(self, sim, pos_mask, neg_mask):
#         # ✅ robust boundary
#         min_pos, max_neg, valid = self._soft_boundary(sim, pos_mask, neg_mask)

#         # MS-style keep masks
#         neg_keep = neg_mask & (sim + self.margin > min_pos.unsqueeze(1))
#         pos_keep = pos_mask & (sim - self.margin < max_neg.unsqueeze(1))

#         # ✅ avoid starving: if empty, fall back to full set for that row
#         # (vectorized row-wise fallback)
#         empty_pos = (pos_keep.sum(dim=1) == 0) & valid
#         empty_neg = (neg_keep.sum(dim=1) == 0) & valid
#         if empty_pos.any():
#             pos_keep[empty_pos] = pos_mask[empty_pos]
#         if empty_neg.any():
#             neg_keep[empty_neg] = neg_mask[empty_neg]

#         return pos_keep, neg_keep, valid

#     def _one_direction(self, sim, pos_mask, neg_mask, delta, thresh):
#         logits = sim / self.tau

#         # ✅ mining
#         pos_keep, neg_keep, valid_anchor = self._ms_mining_masks(sim, pos_mask, neg_mask)
#         pos_keep = pos_keep & valid_anchor.unsqueeze(1)
#         neg_keep = neg_keep & valid_anchor.unsqueeze(1)

#         # ✅ scale_neg curriculum
#         scale_neg = self.scale_neg_start + (self.scale_neg_end - self.scale_neg_start) * delta

#         # MS-guided difficulty scores
#         dp = torch.exp(torch.clamp(-self.scale_pos * (sim - thresh), max=self.clamp_exp))
#         dn = torch.exp(torch.clamp( scale_neg      * (sim - thresh), max=self.clamp_exp))

#         dn_w = dn.detach()
#         dp_w = dp  # keep grad

#         mp = self._masked_mean(dp.detach(), pos_keep, dim=1).unsqueeze(1)
#         mn = self._masked_mean(dn_w,        neg_keep, dim=1).unsqueeze(1)

#         wp = (dp_w / (mp + 1e-12)).clamp_min(1e-12)
#         wn = (dn_w / (mn + 1e-12)).clamp_min(1e-12)

#         cap = self.cap_min + (self.cap_max - self.cap_min) * delta
#         d_pos = delta if self.delta_on_pos else 1.0
#         d_neg = delta

#         log_wp = torch.clamp(d_pos * torch.log(wp), -cap, cap)
#         log_wn = torch.clamp(d_neg * torch.log(wn), -cap, cap)

#         neg_inf = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)

#         log_num = torch.logsumexp(torch.where(pos_keep, logits + log_wp, neg_inf), dim=1)

#         keep_all = pos_keep | neg_keep
#         log_den = torch.logsumexp(
#             torch.where(keep_all, logits + torch.where(pos_keep, log_wp, log_wn), neg_inf),
#             dim=1
#         )

#         valid = (pos_keep.sum(dim=1) > 0) & (neg_keep.sum(dim=1) > 0)
#         loss = -(log_num - log_den)
#         return loss[valid].mean() if valid.any() else sim.new_zeros(())

#     def forward(self, h1, h2, labels, dataset="MSLOSS", epoch=0):
#         if dataset == "cifar10-1":
#             y = torch.argmax(labels, dim=1)
#             pos_mask = (y[:, None] == y[None, :])
#         else:
#             lab = labels.float()
#             pos_mask = (lab @ lab.t()) > 0
#         neg_mask = ~pos_mask

#         B = h1.size(1)
#         sim = (h1 @ h2.t()) / float(B)

#         delta = self._delta(epoch)

#         if self.use_ema_thresh and self.training:
#             self._update_thresh_ema(sim.detach(), pos_mask, neg_mask)
#             thresh = float(self.thresh_ema.item())
#         else:
#             thresh = self.thresh_init

#         i2t = self._one_direction(sim,      pos_mask,      neg_mask,      delta, thresh)
#         t2i = self._one_direction(sim.t(),  pos_mask.t(),  neg_mask.t(),  delta, thresh)
#         return 0.5 * (i2t + t2i)





import torch
import torch.nn as nn

class MSGuidedSelfPacedInfoNCE_V2xSoftMiningGate(nn.Module):
    """
    Keep V2 full-coverage stability, add V3 MS-boundary focus as a *soft boost*.
    - No hard masking in numerator/denominator (large-dataset friendly)
    - Mining uses soft boundary (softmin/softmax) -> robust
    - Boost is self-paced (only opens after gate_start)
    - Neg boost has a cap on sim (avoid pushing extreme hard neg / false neg)
    """

    def __init__(self,
                 temperature=0.2,
                 totalepoch=100,
                 ramp_ratio=1/3,

                 thresh=0.0,
                 scale_pos=2.0,
                 scale_neg=40.0,
                 clamp_exp=50.0,

                 cap_min=2.0,
                 cap_max=6.0,

                 # MS mining gate
                 margin=0.1,
                 gate_start=0.6,     # 前 60% 训练几乎等价 V2，后面再逐步打开 mining boost
                 boost_pos=2.0,      # mined pos 权重最多放大到 2x
                 boost_neg=1.3,      # mined neg 权重放大更保守（大数据集关键）
                 neg_sim_cap=0.2,    # 只 boost sim <= thresh + neg_sim_cap 的 neg（避免最硬 neg）

                 # soft boundary sharpness
                 alpha_pos=8.0,
                 alpha_neg=8.0):
        super().__init__()
        self.tau = temperature
        self.totalepoch = totalepoch
        self.ramp_ratio = ramp_ratio

        self.thresh = thresh
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg
        self.clamp_exp = clamp_exp

        self.cap_min = cap_min
        self.cap_max = cap_max

        self.margin = margin
        self.gate_start = gate_start
        self.boost_pos = boost_pos
        self.boost_neg = boost_neg
        self.neg_sim_cap = neg_sim_cap

        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def _delta(self, epoch: int):
        ramp = max(1, int(self.totalepoch * self.ramp_ratio))
        return min(1.0, float(epoch) / float(ramp))

    def _masked_mean(self, x, mask, dim=1, eps=1e-12):
        mask_f = mask.float()
        s = (x * mask_f).sum(dim=dim)
        c = mask_f.sum(dim=dim).clamp_min(1.0)
        return s / (c + eps)

    def _gate_strength(self, delta: float):
        # 0..1 after gate_start
        if delta <= self.gate_start:
            return 0.0
        return min(1.0, (delta - self.gate_start) / (1.0 - self.gate_start + 1e-12))

    def _soft_boundary(self, sim, pos_mask, neg_mask):
        neg_inf = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)

        # softmin(pos)
        z_pos = torch.where(pos_mask, -self.alpha_pos * sim, neg_inf)
        min_pos = -torch.logsumexp(z_pos, dim=1) / self.alpha_pos

        # softmax(neg)
        z_neg = torch.where(neg_mask,  self.alpha_neg * sim, neg_inf)
        max_neg =  torch.logsumexp(z_neg, dim=1) / self.alpha_neg

        valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        return min_pos, max_neg, valid

    def _one_direction(self, sim, pos_mask, neg_mask, delta):
        logits = sim / self.tau

        # V2 difficulty
        dp = torch.exp(torch.clamp(-self.scale_pos * (sim - self.thresh), max=self.clamp_exp))
        dn = torch.exp(torch.clamp( self.scale_neg * (sim - self.thresh), max=self.clamp_exp))

        dn_w = dn.detach()
        dp_w = dp  # keep grad

        valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)

        # ---- soft mining gate (boost only) ----
        g = self._gate_strength(float(delta))
        if g > 0.0 and valid.any():
            min_pos, max_neg, _ = self._soft_boundary(sim, pos_mask, neg_mask)

            pos_keep = pos_mask & (sim - self.margin < max_neg.unsqueeze(1))
            # neg_keep 同 MS 条件，但加一层 sim cap 防 false neg
            neg_keep = neg_mask & (sim + self.margin > min_pos.unsqueeze(1)) & (sim <= (self.thresh + self.neg_sim_cap))

            # gate factors in [1, boost]
            gp = 1.0 + g * (self.boost_pos - 1.0) * pos_keep.float()
            gn = 1.0 + g * (self.boost_neg - 1.0) * neg_keep.float()

            # normalize with gated values (mean stats stop-grad)
            mp = self._masked_mean((dp.detach() * gp.detach()), pos_mask, dim=1).unsqueeze(1)
            mn = self._masked_mean((dn_w        * gn.detach()), neg_mask, dim=1).unsqueeze(1)

            wp = (dp_w * gp) / (mp + 1e-12)
            wn = (dn_w * gn) / (mn + 1e-12)
            wp = wp.clamp_min(1e-12)
            wn = wn.clamp_min(1e-12)
        else:
            mp = self._masked_mean(dp.detach(), pos_mask, dim=1).unsqueeze(1)
            mn = self._masked_mean(dn_w,        neg_mask, dim=1).unsqueeze(1)
            wp = (dp_w / (mp + 1e-12)).clamp_min(1e-12)
            wn = (dn_w / (mn + 1e-12)).clamp_min(1e-12)

        # V2 self-paced clamp
        cap = self.cap_min + (self.cap_max - self.cap_min) * delta
        log_wp = torch.clamp(delta * torch.log(wp), -cap, cap)
        log_wn = torch.clamp(delta * torch.log(wn), -cap, cap)

        neg_inf = torch.tensor(-1e9, device=sim.device, dtype=sim.dtype)

        logw_pos_only = torch.where(pos_mask, log_wp, neg_inf)
        log_num = torch.logsumexp(logits + logw_pos_only, dim=1)

        logw_all = torch.where(pos_mask, log_wp, torch.where(neg_mask, log_wn, neg_inf))
        log_den = torch.logsumexp(logits + logw_all, dim=1)

        loss = -(log_num - log_den)
        return loss[valid].mean() if valid.any() else sim.new_zeros(())

    def forward(self, h1, h2, labels, dataset="MSLOSS", epoch=0):
        if dataset == "cifar10-1":
            y = torch.argmax(labels, dim=1)
            pos_mask = (y[:, None] == y[None, :])
        else:
            lab = labels.float()
            pos_mask = (lab @ lab.t()) > 0
        neg_mask = ~pos_mask

        B = h1.size(1)
        sim = (h1 @ h2.t()) / float(B)

        delta = self._delta(epoch)
        i2t = self._one_direction(sim,        pos_mask,       neg_mask,       delta)
        t2i = self._one_direction(sim.t(),    pos_mask.t(),   neg_mask.t(),   delta)
        return 0.5 * (i2t + t2i)
