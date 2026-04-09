import torch
import torch.nn as nn
import torch.nn.functional as F

class HardNegCLIPInfoNCE_Pro_NoLabel(nn.Module):
    """
    Label-free upgrade of HardNegCLIPInfoNCE:
    1) hard + random negatives (more stable than pure topk)
    2) learnable logit_scale (CLIP-style), with clamp
    3) curriculum: gradually increase hard ratio by epoch
    """
    def __init__(self,
                 tau_init=0.07,
                 k_hard=32,
                 k_rand=32,
                 symmetric=True,
                 learnable_scale=True,
                 scale_max=100.0,
                 use_curriculum=True,
                 curr_start_epoch=5,
                 curr_warm_epochs=10,
                 min_valid_negs=8):
        super().__init__()
        self.k_hard = k_hard
        self.k_rand = k_rand
        self.symmetric = symmetric

        self.learnable_scale = learnable_scale
        self.scale_max = scale_max

        init = torch.log(torch.tensor(1.0 / tau_init))
        self.logit_scale = nn.Parameter(init) if learnable_scale else None
        self.tau_fixed = tau_init

        self.use_curriculum = use_curriculum
        self.curr_start_epoch = curr_start_epoch
        self.curr_warm_epochs = max(1, curr_warm_epochs)
        self.min_valid_negs = min_valid_negs

        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def forward(self, h_img, h_txt):
        a = F.normalize(h_img, dim=1)
        b = F.normalize(h_txt, dim=1)
        sim = a @ b.t()  # [B,B]

        loss_i2t = self._dir(sim)
        if not self.symmetric:
            return loss_i2t
        loss_t2i = self._dir(sim.t())
        return 0.5 * (loss_i2t + loss_t2i)

    def _get_scale(self, device):
        if self.learnable_scale:
            return torch.exp(self.logit_scale).clamp(max=self.scale_max).to(device)
        return torch.tensor(1.0 / max(self.tau_fixed, 1e-6), device=device)

    def _hard_ratio(self):
        if not self.use_curriculum:
            return 1.0
        if self._epoch < self.curr_start_epoch:
            return 0.0
        t = (self._epoch - self.curr_start_epoch) / float(self.curr_warm_epochs)
        return float(max(0.0, min(1.0, t)))

    def _dir(self, sim):
        B = sim.size(0)
        device = sim.device
        neg_fill = -1e4  # fp16 safe
        diag = torch.eye(B, device=device, dtype=torch.bool)

        pos = sim[diag]                            # [B]
        sim_neg = sim.masked_fill(diag, neg_fill)  # [B,B]

        k_h = min(self.k_hard, B - 1)
        k_r = min(self.k_rand, B - 1 - k_h)

        # curriculum: early fewer hard negs, later more hard
        hr = self._hard_ratio()
        k_h_eff = int(round(k_h * hr))
        k_r_eff = (k_h + k_r) - k_h_eff  # keep total neg count roughly constant

        # hard negatives
        if k_h_eff > 0:
            hard_vals, hard_idx = torch.topk(sim_neg, k=k_h_eff, dim=1, largest=True)  # [B,k_h_eff]
        else:
            hard_vals = sim_neg.new_empty((B, 0))
            hard_idx = None
        # random negatives (exclude diag, and exclude selected hard indices)
        if k_r_eff > 0:
            rand_vals = []
            valid = (~diag).clone()  # [B,B]
            if hard_idx is not None:
                valid.scatter_(1, hard_idx, False)

            for i in range(B):
                idx = torch.nonzero(valid[i], as_tuple=False).view(-1)
                if idx.numel() < self.min_valid_negs:
                    idx = torch.nonzero(~diag[i], as_tuple=False).view(-1)

                perm = idx[torch.randperm(idx.numel(), device=device)]
                pick = perm[:k_r_eff] if perm.numel() >= k_r_eff else perm
                vals = sim[i, pick]
                if vals.numel() < k_r_eff:
                    vals = torch.cat([vals, sim_neg.new_full((k_r_eff - vals.numel(),), neg_fill)], dim=0)
                rand_vals.append(vals)
            rand_vals = torch.stack(rand_vals, dim=0)  # [B,k_r_eff]
        else:
            rand_vals = sim_neg.new_empty((B, 0))

        negs = torch.cat([hard_vals, rand_vals], dim=1)  # [B, Kneg]

        scale = self._get_scale(device)
        logits = torch.cat([pos.unsqueeze(1), negs], dim=1) * scale  # [B,1+Kneg]

        targets = torch.zeros(B, device=device, dtype=torch.long)
        return F.cross_entropy(logits, targets)
