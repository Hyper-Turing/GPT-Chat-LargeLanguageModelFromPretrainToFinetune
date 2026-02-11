"""
基于nanochat/gpt 改动适配Qwen2.5
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================================
# Config
# ======================================================================

@dataclass
class GPTConfig:
    max_seq_len: int = 2048
    vocab_size: int = 152064
    n_layer: int = 28               # Qwen2.5-7B
    n_head: int = 28                # Q heads
    n_kv_head: int = 4              # KV heads (GQA)
    n_embd: int = 3584
    intermediate_size: int = 18944
    rope_base: float = 1000000.0 
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0

    @classmethod
    def qwen2_5_0_7b(cls):
        return cls(n_layer=28, n_head=28, n_kv_head=4, n_embd=3584,
                   intermediate_size=18944, vocab_size=152064)
    

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # 在 float32 下计算归一化，避免 bf16 精度问题
        input_dtype = x.dtype
        x = x.float()
        x = F.rms_norm(x, (x.size(-1),), self.weight.float(), self.eps)
        return x.to(input_dtype)
    

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # 在 float32 下计算旋转，避免 bf16 精度损失
    cos = cos.float()
    sin = sin.float()
    y1 = x1.float() * cos - x2.float() * sin
    y2 = x1.float() * sin + x2.float() * cos
    return torch.cat([y1, y2], 3).to(x.dtype)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.n_rep = self.n_head // self.n_kv_head

        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=True)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=True)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)  # o_proj 无 bias

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _repeat_kv(self, x):
        if self.n_rep == 1:
            return x
        B, T, n_kv, D = x.shape
        return (x[:, :, :, None, :]
                .expand(B, T, n_kv, self.n_rep, D)
                .reshape(B, T, self.n_head, D))
    
    def forward(self, x, cos, sin):
        B, T, C = x.size()

        # QKV 投影
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # GQA: 复制 KV heads
        k = self._repeat_kv(k)  # (B, T, n_head, head_dim)
        v = self._repeat_kv(v)

        # 转置为 (B, n_head, T, head_dim) 用于 attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, 
                                           dropout_p=self.attn_dropout.p if self.training else 0.0)
        
        # 输出投影
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.n_embd, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.n_embd, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.ln_2 = RMSNorm(config.n_embd, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        # Pre-LN + 残差
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.ln_f = RMSNorm(config.n_embd, config.rms_norm_eps)

        # RoPE 预计算
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary(config.max_seq_len, head_dim, config.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=1000000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()[None, :, None, :]   # (1, seq_len, 1, head_dim/2)
        sin = freqs.sin()[None, :, None, :]
        return cos, sin
    
    def forward(self, idx, targets=None):
        """
          wte → LN → [Block x n_layer] → ln_f → lm_head
        """
        B, T = idx.size()

        # Token embedding
        x = self.transformer.wte(idx)

        # RoPE slice
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, cos, sin)

        # Final norm + LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            # 语言模型需要 shift: logits[t] 预测 targets[t+1]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        else:
            # 推理：只取最后一个位置
            return logits[:, [-1], :], None
        

    @torch.no_grad()
    def generate(
        self, 
        idx, 
        max_new_tokens, 
        temperature=1.0, 
        top_k=None, 
        top_p=None,
        do_sample=True,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """
        生成文本序列
        
        Args:
            idx: 输入 token ids, shape [B, T]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: top-k 采样
            top_p: nucleus 采样
            do_sample: 是否采样（False 则贪心解码）
            eos_token_id: 停止 token（可以是 int 或 list）
            pad_token_id: padding token id
        """
        B = idx.size(0)
        
        # 支持多个 EOS token
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=idx.device)
        
        # 跟踪哪些序列已经结束
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)

        for _ in range(max_new_tokens):
            # 如果所有序列都结束了，提前退出
            if finished.all():
                break
                
            # 截断到最大序列长度
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 取最后一个位置的 logits
            
            # 对于已经结束的序列，不再生成（logits 置为极小值，除了 pad_token）
            if finished.any() and pad_token_id is not None:
                logits[finished] = -float('Inf')
                logits[finished, pad_token_id] = 0
            
            # 应用 temperature
            if do_sample and temperature > 0:
                logits = logits / temperature
            
            # Top-k 采样
            if do_sample and top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            
            # Top-p (nucleus) 采样
            if do_sample and top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的 token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                # 将需要移除的 token 的 logits 设为 -inf
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # 采样或贪心
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            # 拼接新 token
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 检查是否生成了 EOS token
            if eos_token_id is not None:
                # 检查新生成的 token 是否在 eos_token_id 列表中
                is_eos = (idx_next.squeeze(1).unsqueeze(1) == eos_token_id.unsqueeze(0)).any(dim=1)
                finished = finished | is_eos

        return idx
    
    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen2.5-7B"):
        """
        从 HuggingFace 加载预训练权重:
          HF                                    → GPT
          model.embed_tokens.weight             → transformer.wte.weight
          model.layers.{i}.self_attn.q_proj.*   → transformer.h.{i}.attn.c_q.*
          model.layers.{i}.self_attn.k_proj.*   → transformer.h.{i}.attn.c_k.*
          model.layers.{i}.self_attn.v_proj.*   → transformer.h.{i}.attn.c_v.*
          model.layers.{i}.self_attn.o_proj.*   → transformer.h.{i}.attn.c_proj.*
          model.layers.{i}.mlp.gate_proj.*      → transformer.h.{i}.mlp.gate_proj.*
          model.layers.{i}.mlp.up_proj.*        → transformer.h.{i}.mlp.up_proj.*
          model.layers.{i}.mlp.down_proj.*      → transformer.h.{i}.mlp.down_proj.*
          model.layers.{i}.input_layernorm.*    → transformer.h.{i}.ln_1.*
          model.layers.{i}.post_attention_layernorm.* → transformer.h.{i}.ln_2.*
          model.norm.weight                     → ln_f.weight
          lm_head.weight                        → lm_head.weight
        """
        from transformers import AutoModelForCausalLM, AutoConfig

        # print(f"Loading HF model: {model_name}")
        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # rope_base: 兼容不同版本的 HF config 属性名
        # print(vars(hf_config))  # 查看 HF config 所有属性
        rope_base = getattr(hf_config, 'rope_theta', None) \
                    or getattr(hf_config, 'rope_base', None) \
                    or 1000000.0
        # print(f"  rope_base = {rope_base}")

        # 从 HF config 构建我们的 config
        config = GPTConfig(
            max_seq_len=min(hf_config.max_position_embeddings, 2048),  # 限制长度省显存
            vocab_size=hf_config.vocab_size,
            n_layer=hf_config.num_hidden_layers,
            n_head=hf_config.num_attention_heads,
            n_kv_head=hf_config.num_key_value_heads,
            n_embd=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            rope_base=rope_base,
            rms_norm_eps=hf_config.rms_norm_eps,
        )
        # print(f"Config: {config}")

        # 创建我们的模型（随机初始化）
        model = cls(config)

        # 加载 HF 权重
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True,
        )
        hf_sd = hf_model.state_dict()

        # ---- 键名映射 ----
        mapping = {
            "model.embed_tokens.weight": "transformer.wte.weight",
            "model.norm.weight": "ln_f.weight",
            "lm_head.weight": "lm_head.weight",
        }
        for i in range(config.n_layer):
            hf_pre = f"model.layers.{i}"
            my_pre = f"transformer.h.{i}"
            layer_map = {
                f"{hf_pre}.self_attn.q_proj.weight": f"{my_pre}.attn.c_q.weight",
                f"{hf_pre}.self_attn.q_proj.bias":   f"{my_pre}.attn.c_q.bias",
                f"{hf_pre}.self_attn.k_proj.weight": f"{my_pre}.attn.c_k.weight",
                f"{hf_pre}.self_attn.k_proj.bias":   f"{my_pre}.attn.c_k.bias",
                f"{hf_pre}.self_attn.v_proj.weight": f"{my_pre}.attn.c_v.weight",
                f"{hf_pre}.self_attn.v_proj.bias":   f"{my_pre}.attn.c_v.bias",
                f"{hf_pre}.self_attn.o_proj.weight": f"{my_pre}.attn.c_proj.weight",
                f"{hf_pre}.mlp.gate_proj.weight":    f"{my_pre}.mlp.gate_proj.weight",
                f"{hf_pre}.mlp.up_proj.weight":      f"{my_pre}.mlp.up_proj.weight",
                f"{hf_pre}.mlp.down_proj.weight":    f"{my_pre}.mlp.down_proj.weight",
                f"{hf_pre}.input_layernorm.weight":           f"{my_pre}.ln_1.weight",
                f"{hf_pre}.post_attention_layernorm.weight":  f"{my_pre}.ln_2.weight",
            }
            mapping.update(layer_map)

        sd = model.state_dict()
        for hf_key, my_key in mapping.items():
            if hf_key in hf_sd:
                assert hf_sd[hf_key].shape == sd[my_key].shape, \
                    f"Shape mismatch: {hf_key} {hf_sd[hf_key].shape} vs {my_key} {sd[my_key].shape}"
                sd[my_key].copy_(hf_sd[hf_key])
            elif hf_key == "lm_head.weight" and getattr(hf_config, 'tie_word_embeddings', False):
                # lm_head 与 embed_tokens 权重共享
                print(f"  tie_word_embeddings: lm_head.weight ← model.embed_tokens.weight")
                sd[my_key].copy_(hf_sd["model.embed_tokens.weight"])
            else:
                print(f"  Warning: HF key not found: {hf_key}")

        model.load_state_dict(sd)
        del hf_model, hf_sd  # 释放 HF 模型内存
        print("Weights loaded successfully!")
        return model