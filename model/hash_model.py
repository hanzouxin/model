import os
import torch
import logging
import torch.nn as nn
import numpy as np
import math
from typing import Union
# from model.swin_model import SwinTransformer
from model.model import build_model
from utils import get_logger, get_summary_writer
from model.onem import CrossModalHGNN
from model.prompt_generator import PromptGenerator,get_class_info,tokenize_clip_texts,build_oracle_prompt_texts
from model.simple_tokenizer import SimpleTokenizer as Tokenizer
import torch.nn.functional as F

class VisualAdapter(nn.Module):
    def __init__(self, dim, bottleneck=64, dropout=0.1, init_alpha=1e-2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim)

        # 关键：让它一开始几乎是 identity
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(self, x):
        residual = self.norm(x)
        residual = self.down(residual)
        residual = self.act(residual)
        residual = self.drop(residual)
        residual = self.up(residual)
        return x + self.alpha * residual

class GlobalTopKAGP(nn.Module):
    """
    全局版 AGP：batch 级对齐池化（用全局特征做）
    - S = cos(gI, gT) -> [B,B]
    - 每个图像从 top-k 文本聚合消息；每个文本从 top-k 图像聚合消息
    """
    def __init__(self, dim=512, tau=0.2, topk=8, drop=0.0,
                 detach_affinity=True, alpha=0.3, use_proj=True):
        super().__init__()
        self.tau = tau
        self.topk = topk
        self.drop = nn.Dropout(drop)
        self.detach_affinity = detach_affinity
        self.alpha = alpha

        self.proj_i = nn.Linear(dim, dim, bias=False) if use_proj else nn.Identity()
        self.proj_t = nn.Linear(dim, dim, bias=False) if use_proj else nn.Identity()
        self.ln_i = nn.LayerNorm(dim)
        self.ln_t = nn.LayerNorm(dim)

    def _row_topk_softmax(self, S: torch.Tensor, k: int):
        B, Bt = S.shape
        if k <= 0 or k >= Bt:
            return torch.softmax(S, dim=1)

        neg_large = -1e4 if S.dtype in (torch.float16, torch.bfloat16) else -1e9
        idx = S.topk(k, dim=1).indices
        mask = torch.zeros_like(S, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        S_masked = S.masked_fill(~mask, neg_large)
        return torch.softmax(S_masked, dim=1)

    def forward(self, gI: torch.Tensor, gT: torch.Tensor):
        """
        gI,gT: [B,D]
        return: gI2,gT2, S
        """
        gi = F.normalize(gI, dim=-1)
        gt = F.normalize(gT, dim=-1)
        S = (gi @ gt.t()) / self.tau  # [B,B]
        if self.detach_affinity:
            S = S.detach()

        # image <- texts
        A_it = self._row_topk_softmax(S, self.topk)  # [B,B]
        A_it = self.drop(A_it)
        msg_i = A_it @ self.proj_t(gT)  # [B,D]

        # text <- images
        A_ti = self._row_topk_softmax(S.t(), self.topk)  # [B,B]
        A_ti = self.drop(A_ti)
        msg_t = A_ti @ self.proj_i(gI)  # [B,D]

        gI2 = self.ln_i(gI + self.alpha * msg_i)
        gT2 = self.ln_t(gT + self.alpha * msg_t)
        return gI2, gT2, S


class BatchBanzhafInteraction(nn.Module):
    """
    Batch-level Banzhaf on S: [B,B] -> I: [B,B]
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, S: torch.Tensor):
        B, Bt = S.shape
        assert B == Bt and B >= 2, "Need square S with B>=2"

        top2_vals, top2_idx = torch.topk(S, k=2, dim=1)  # [B,2]
        m1 = top2_vals[:, 0]   # [B]
        idx1 = top2_idx[:, 0]  # [B]
        m2 = top2_vals[:, 1]   # [B]
        m2 = torch.where(torch.isfinite(m2), m2, m1)

        S_bar = m1.mean()  # scalar

        sum_m1 = m1.sum()
        S_minus_i = (sum_m1 - m1) / (B - 1)  # [B]

        j_idx = torch.arange(B, device=S.device)
        hit = (idx1.unsqueeze(1) == j_idx.unsqueeze(0))  # [B,B]
        m_minus_j = m1.unsqueeze(1).expand(B, B)
        m_minus_j = torch.where(hit, m2.unsqueeze(1), m_minus_j)
        S_minus_j = m_minus_j.mean(dim=0)  # [B]

        sum_m_minus_j = m_minus_j.sum(dim=0, keepdim=True)  # [1,B]
        S_minus_ij = (sum_m_minus_j - m_minus_j) / (B - 1)  # [B,B]

        I = S_bar - S_minus_i.unsqueeze(1) - S_minus_j.unsqueeze(0) + S_minus_ij
        return I


class BanzhafGuidance(nn.Module):
    """
    用 Banzhaf 产生样本权重 w（只引导，不做强 loss）
    """
    def __init__(self, tau=0.2, eps=1e-6):
        super().__init__()
        self.tau = tau
        self.eps = eps
        self.bz = BatchBanzhafInteraction(eps=eps)

    def forward(self, gI, gT):
        gi = F.normalize(gI, dim=-1)
        gt = F.normalize(gT, dim=-1)
        S = gi @ gt.t()  # [B,B]

        with torch.no_grad():
            I = self.bz(S.detach())
            pos = torch.diagonal(I, 0)  # [B]
            pos = torch.clamp(pos, min=-10.0, max=10.0)
            w = F.softmax(pos / self.tau, dim=0)  # [B], sum=1

            # hard negative (可选)
            eye = torch.eye(S.size(0), device=S.device, dtype=torch.bool)
            I_neg = I.masked_fill(eye, -1e9)
            hard_j = torch.argmax(I_neg, dim=1)  # [B]

        return w, S, I, hard_j


def banzhaf_weighted_infonce(gI, gT, w, tau=0.2, eps=1e-6):
    """
    可选：把 w 用到一个很轻的全局对比 loss（稳定、好用）
    gI,gT: [B,D], w: [B] sum=1
    """
    gi = F.normalize(gI, dim=-1)
    gt = F.normalize(gT, dim=-1)
    logits = gi @ gt.t() / tau
    target = torch.arange(logits.size(0), device=logits.device)

    li = F.cross_entropy(logits, target, reduction="none")      # [B]
    lt = F.cross_entropy(logits.t(), target, reduction="none")  # [B]

    w = w / (w.sum() + eps)
    return (w * li).sum() + (w * lt).sum()

class CrossAttention(nn.Module):
    def __init__(self, dim_out, num_heads=8):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True)
        self.output_linear = nn.Linear(dim_out, dim_out)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        # Add an extra dimension for sequence length (which is 1 in this case)
        query = query.unsqueeze(1)  # [B, 1, dim_out]
        key = key.unsqueeze(1)      # [B, 1, dim_out]
        value = value.unsqueeze(1)  # [B, 1, dim_out]

        # Multihead Attention
        attn_output, attn_weights = self.multihead_attn(query, key, value)  # attn_output: [B, 1, dim_out]

        # Output linear transformation
        attn_output = self.output_linear(attn_output)

        # Residual connection and Layer Normalization
        attn_output = self.layer_norm(attn_output + query)

        # Squeeze the sequence dimension
        output = attn_output.squeeze(1)  # [B, dim_out]

        return output

class GatedFusion(nn.Module):
    def __init__(self, dim_out, dropout=0.1):
        super(GatedFusion, self).__init__()

        self.main_proj = nn.Linear(dim_out, dim_out)
        self.prompt_proj = nn.Linear(dim_out, dim_out)

        self.gate_net = nn.Sequential(
            nn.Linear(dim_out * 2, dim_out),
            nn.Sigmoid()
        )

        self.output_linear = nn.Linear(dim_out, dim_out)
        self.layer_norm = nn.LayerNorm(dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, main_feat, prompt_feat):
        """
        main_feat:   [B, dim_out]
        prompt_feat: [B, dim_out]
        return:      [B, dim_out]
        """

        main_h = self.main_proj(main_feat)       # [B, D]
        prompt_h = self.prompt_proj(prompt_feat) # [B, D]

        gate = self.gate_net(torch.cat([main_feat, prompt_feat], dim=-1))  # [B, D]

        fused = main_h + gate * prompt_h
        fused = self.output_linear(fused)
        fused = self.dropout(fused)

        out = self.layer_norm(main_feat + fused)
        return out


class TextPromptGraphFusion(nn.Module):

    def __init__(self, dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        self.q_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.k_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.v_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

        self.out_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])

        self.ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim)
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # 最后对两个节点做加权聚合
        self.node_score = nn.Linear(dim, 1)

    def forward(self, text_feat, prompt_feat, prompt_conf=None):
        """
        text_feat:   [B, D]
        prompt_feat: [B, D]
        prompt_conf: [B] or [B,1], optional
        """
        if prompt_conf is not None:
            if prompt_conf.dim() == 1:
                prompt_conf = prompt_conf.unsqueeze(-1)   # [B,1]
            prompt_feat = prompt_feat * prompt_conf

        # 两个节点：text / prompt
        x = torch.stack([text_feat, prompt_feat], dim=1)   # [B, 2, D]

        for i in range(self.num_layers):
            q = self.q_proj[i](x)   # [B, 2, D]
            k = self.k_proj[i](x)   # [B, 2, D]
            v = self.v_proj[i](x)   # [B, 2, D]

            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim)   # [B,2,2]
            attn = torch.softmax(attn, dim=-1)

            msg = torch.matmul(attn, v)   # [B,2,D]
            msg = self.dropout(self.out_proj[i](msg))

            x = self.norm1[i](x + msg)
            x = self.norm2[i](x + self.dropout(self.ffn[i](x)))

        score = self.node_score(x).squeeze(-1)   # [B,2]
        node_weights = torch.softmax(score, dim=-1)

        fused_feat = torch.sum(x * node_weights.unsqueeze(-1), dim=1)   # [B,D]

        return fused_feat

class LinearHash(nn.Module):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        # self.fc.apply(weights_init_kaiming)
        self.drop_out = nn.Dropout(p=0.2)
    
    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))

def build_filtered_prompt_texts(topk_indices, label, class_names, label_map=None):

    topk_indices = topk_indices.detach().cpu()
    label = label.detach().cpu()

    prompt_texts = []

    for i in range(label.size(0)):
        true_idx = set(torch.nonzero(label[i] > 0, as_tuple=False).squeeze(-1).tolist())

        pred_idx = []
        for idx in topk_indices[i].tolist():
            if idx >= 0 and idx in true_idx:
                pred_idx.append(idx)

        class_texts = []
        for idx in pred_idx:
            name = class_names[idx]
            if label_map is not None and name in label_map:
                name = label_map[name]
            else:
                name = name.replace("_", " ")
            class_texts.append(name)

        if len(class_texts) == 0:
            prompt_text = ""
        else:
            prompt_text = " ".join(class_texts)

        prompt_texts.append(prompt_text)

    return prompt_texts

class DCMHT(nn.Module):

    def __init__(self, 
                outputDim=64, 
                clipPath="./ViT-B-32.pt", 
                writer=None, 
                saveDir="./result/log", 
                logger: logging.Logger=None, 
                is_train=True):
        super(DCMHT, self).__init__()
        os.makedirs(saveDir, exist_ok=True)
        self.logger = logger if logger is not None else get_logger(os.path.join(saveDir, "train.log" if is_train else "test.log"))
        self.writer = writer if writer is not None and is_train else get_summary_writer(os.path.join(saveDir, "tensorboard"))
        embedDim, self.clip = self.load_clip(clipPath)
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        self.clip = self.clip.to(self.device)
        self.clip_tokenizer = Tokenizer()
        self.dataset_name = "nuswide"
        self.class_names, self.label_map = get_class_info("nuswide")
        
        self.use_oracle_prompt = False     # 先开着做诊断实验
        self.use_filtered_prompt = True
        self.oracle_topk = None              # 先和你当前 topk 保持一致
        self.cross_attention = CrossAttention(512,8).to(self.device)
        self.prompt_fusion = GatedFusion(embedDim, dropout=0.1)
        self.text_prompt_fusion = TextPromptGraphFusion(dim=embedDim, num_layers=2,dropout=0.1)
        self.prompt_generator = PromptGenerator(
            dataset_name="nuswide",
            classifier_ckpt="./checkpoints/prompt_cls_nuswide.pt",
            bert_path="/home/yuck/bert-base-uncased",
            device=self.device,
            maxWords=32,
            prob_threshold=0.7,
        )
        self.image_adapter = VisualAdapter(dim=embedDim, bottleneck=64, dropout=0.1, init_alpha=1e-2)
        self.gcr = CrossModalHGNN()
        self.agp = GlobalTopKAGP(dim=embedDim, tau=0.2, topk=8, detach_affinity=True, alpha=0.3)
        self.bz_guidance = BanzhafGuidance(tau=0.2)

        self.image_hash = LinearHash(inputDim=embedDim, outputDim=outputDim)
        self.text_hash = LinearHash(inputDim=embedDim, outputDim=outputDim)

    def freezen(self):
        for name, param in self.clip.named_parameters():
            # print(name)
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                                        or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                # print("1")
                continue
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= 12:
                    # print("2")
                    continue
            if name.find("conv2.") == 0:
                # print("3")
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")

        return state_dict["text_projection"].shape[1], build_model(state_dict)
    
    def train(self, mode=True):
        super().train(mode)
        return self

    def eval(self):
        return self.train(False)


    def encoding(self, image,text,raw_text=None,label=None,return_extra=False):

        image_global, image_local = self.clip.encode_image(image,return_tokens=True) #512
        image_global=self.image_adapter(image_global)
        text_global, text_local, eot_indices = self.clip.encode_text(text,return_tokens=True)
        if self.use_oracle_prompt:
            if label is None:
                raise ValueError("use_oracle_prompt=True 时，encoding必须传入label")
            prompt_texts = build_oracle_prompt_texts(
                label=label,
                class_names=self.class_names,
                label_map=self.label_map,
                topk=self.oracle_topk
            )

            prompt_ids = tokenize_clip_texts(
                texts=prompt_texts,
                clip_tokenizer=self.prompt_generator.clip_tokenizer,
                maxWords=32
            ).to(self.device)

            topk_indices = None
            probs = None
        else:
            prompt_ids, topk_indices,selected_counts, probs, prompt_texts = self.prompt_generator(raw_text)

            if self.use_filtered_prompt:
                if label is None:
                    raise ValueError("use_filtered_prompt=True 时，encoding必须传入label")

                prompt_texts = build_filtered_prompt_texts(
                    topk_indices=topk_indices,
                    label=label,
                    class_names=self.class_names,
                    label_map=self.label_map
                )

                prompt_ids = tokenize_clip_texts(
                    texts=prompt_texts,
                    clip_tokenizer=self.prompt_generator.clip_tokenizer,
                    maxWords=32
                ).to(self.device)

        prompt_global, prompt_tokens, _ = self.clip.encode_text(prompt_ids, return_tokens=True)

        # prompt_texts_T = build_oracle_prompt_texts(
        #         label=label,
        #         class_names=self.class_names,
        #         label_map=self.label_map,
        #         topk=self.oracle_topk
        #     )
        # print("raw_text example:", raw_text[0])
        # print("True prompt example:", prompt_texts_T[0])
        # print("pre prompt example:", prompt_texts[0])

        # fused_text_global = self.cross_attention(text_global, prompt_global, prompt_global)
        fused_text_global = self.text_prompt_fusion(text_global, prompt_global)
        image_global = image_global / (image_global.norm(dim=-1, keepdim=True) + 1e-6)
        fused_text_global = fused_text_global / (fused_text_global.norm(dim=-1, keepdim=True) + 1e-6)

        
        gI1, gT1 = self.gcr(image_global, fused_text_global)
        gI2, gT2, S = self.agp(gI1, gT1)
        w, S2, I, hard_j = self.bz_guidance(gI2, gT2)
        image_hash = self.image_hash(gI2)
        text_hash = self.text_hash(gT2)
    
        if return_extra:
            extra = {"w": w, "S": S2, "I": I, "hard_j": hard_j, "gI": gI2, "gT": gT2,}
            return image_hash, text_hash,extra
        
        return image_hash,text_hash
    
    def forward(self, image, text,label,raw_text):
        image_hash,text_hash, extra = self.encoding(image,text,raw_text,label,return_extra=True)
        align_loss = banzhaf_weighted_infonce(extra["gI"], extra["gT"], extra["w"], tau=0.2)
        return image_hash, text_hash,align_loss








