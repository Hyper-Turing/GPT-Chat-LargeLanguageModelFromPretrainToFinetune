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
    # 工厂模式
    def qwen2_5_0_7b(cls):
        return cls(n_layer=28, n_head=28, n_kv_head=4, n_embd=3584,
                   intermediate_size=18944, vocab_size=152064)

"""
为什么kv_head=4? 
Attention中kv所占显存=2 * batch_size * kv_head * seq_len * head_dim * n_layers * 2(float16)
MHA 中 kv_head = n_head = 28
GQA 中 kv_head = 4
两者差距 7 倍，显存占用大幅降低
"""


# ======================================================================
# KVCache 
# ======================================================================

class KVCache:
    """
    KV 缓存类，适配 PyTorch 原生 scaled_dot_product_attention。
    核心设计:
    - 预分配缓存张量，避免动态分配开销
    - 通过 seq_len 追踪当前已填充的位置
    - 支持 prefill (预填充) 和逐 token decode 两种模式
    """
    def __init__(self, batch_size, num_kv_heads, max_seq_len, head_dim, num_layers, device, dtype):
        """
        Args:
            batch_size:   批次大小
            num_kv_heads: KV 头数 (GQA 中 n_kv_head)
            max_seq_len:  最大序列长度 (缓存容量)
            head_dim:     每个注意力头的维度
            num_layers:   Transformer 层数
            device:       设备
            dtype:        数据类型
        """
        self.batch_size = batch_size
        self.n_kv_heads = num_kv_heads
        self.n_layers = num_layers
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim

        # 预分配 (n_layers, B, H, T, D)
        # Transformer中每一层都有独立的kv
        # 统一使用同一个KVCache对象管理所有层的缓存，而不是每层单独创建对象
        # 注意 H 和 T 的顺序
        self.k_cache = torch.zeros(
            num_layers, batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            num_layers, batch_size, num_kv_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.seq_len = 0 # 尾指针

    def reset(self):
        """重置缓存"""
        self.seq_len = 0

    def get_seq_len(self):
        """获取当前缓存中的 token 数"""
        return self.seq_len
    
    def update(self, layer_idx, k_new, v_new):
        """
        更新指定层的 KV 缓存并返回完整 KV。

        Args:
            layer_idx: 层索引
            k_new: 新的 K, shape (B, n_kv_head, T_new, head_dim)
            v_new: 新的 V, shape (B, n_kv_head, T_new, head_dim)

        Returns:
            k_full: (B, n_kv_head, seq_len + T_new, head_dim)
            v_full: (B, n_kv_head, seq_len + T_new, head_dim)
        """
        T_new = k_new.size(2) # 新token的数量，prefill时>1
        start = self.seq_len
        end = start + T_new

        assert end <= self.max_seq_len, f"KV Cache overflow: {end} > {self.max_seq_len}"

        # 写入新的KV到huanc
        self.k_cache[layer_idx, :, :, start:end, :] = k_new
        self.v_cache[layer_idx, :, :, start:end, :] = v_new

        # 返回完整的 KV (从 0 到 end)
        k_full = self.k_cache[layer_idx, :, :, :end, :]
        v_full = self.v_cache[layer_idx, :, :, :end, :]
        return k_full, v_full
    

    def advance(self, num_tokens):
        """推进缓存位置"""
        self.seq_len += num_tokens

    def prefill(self, other):
        """
        用于批量生成场景:为同一个问题生成多个答案时
        先 batch=1 预填充, 再复制到 batch=N 的缓存中
        """
        assert self.seq_len == 0, "Cannot prefill a non-empty KV cache"
        pos = other.seq_len
        # 广播: other 可能是 batch=1, self 可能是 batch=N
        self.k_cache[:, :, :, :pos, :] = other.k_cache[:, :, :, :pos, :]
        self.v_cache[:, :, :, :pos, :] = other.v_cache[:, :, :, :pos, :]
        self.seq_len = pos

# ======================================================================
# Modules
# ======================================================================

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
        """
        将 KV heads 复制以匹配 Q heads 数量 (GQA)
        x: (B, n_kv_head, T, head_dim)
        return: (B, n_head, T, head_dim)
        """
        if self.n_rep == 1:
            return x
        B, n_kv, T, D = x.shape
        return (x[:, :, None, :, :]
            .expand(B, n_kv, self.n_rep, T, D)
            .reshape(B, self.n_head, T, D))
    
    def forward(self, x, cos, sin, kv_cache: Optional[KVCache] = None):
        B, T, C = x.size()

        # QKV 投影
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # 转置为 (B, n_head, T, head_dim) 用于 attention
        q = q.transpose(1, 2) # (B, n_head, T, head_dim)
        k = k.transpose(1, 2) # (B, n_kv_head, T, head_dim)
        v = v.transpose(1, 2)

        # KVCache
        if kv_cache is not None:
            # 更新缓存并获取完整的 kv
            k, v = kv_cache.update(self.layer_idx, k, v) # k, v 现在是 (B, n_kv_head, total_seq_len, head_dim)

        # GQA: 复制 KV heads  以匹配 Q heads
        k = self._repeat_kv(k)  # (B, n_head, T, head_dim)
        v = self._repeat_kv(v)
        
        # Attention
        # 当使用 KV cache 时, Q 可能只有 1 个 token, 而 KV 有很多
        # is_causal=True 只在 Q 和 KV 长度相同时正确
        # 当 decode (T_q=1) 时, 不需要 causal mask (单 token 天然 causal)
        is_causal = (q.size(2) == k.size(2))
        y = F.scaled_dot_product_attention(q, k, v, 
                                           is_causal=is_causal, 
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

    def forward(self, x, cos, sin, kv_cache: Optional[KVCache] = None):
        # Pre-LN + 残差
        x = x + self.attn(self.ln_1(x), cos, sin, kv_cache=kv_cache)
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
    
    def forward(self, idx, targets=None, kv_cache: Optional[KVCache] = None):
        """
        wte → LN → [Block x n_layer] → ln_f → lm_head
        支持KV Cache
        """
        B, T = idx.size()

        # Token embedding
        x = self.transformer.wte(idx)

        # RoPE slice
        # 需要根据 KV Cahce 中的位置来取得正确的 cos/sin
        if kv_cache is not None:
            pos_start = kv_cache.get_seq_len()
            cos = self.cos[:, pos_start:pos_start + T]
            sin = self.sin[:, pos_start:pos_start + T]
        else:
            cos = self.cos[:, :T]
            sin = self.sin[:, :T]

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x, cos, sin, kv_cache=kv_cache)

        # 所有层完成后统一推进缓存位置
        if kv_cache is not None:
            kv_cache.advance(T)

        # Final norm + LM head
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            return logits, loss
        else:
            # 推理：只取最后一个位置
            return logits[:, [-1], :], None
        
    def _create_kv_cache(self, batch_size, seq_len, device, dtype):
        return KVCache(
            batch_size=batch_size,
            num_kv_heads=self.config.n_kv_head,
            max_seq_len=seq_len,
            head_dim=self.config.n_embd // self.config.n_head,
            num_layers=self.config.n_layer,
            device=device,
            dtype=dtype,
        )


    @torch.no_grad()
    def generate(
        self, 
        idx, 
        max_new_tokens, 
        temperature=1.0, 
        top_k=None, 
        top_p=None,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=None,
        pad_token_id=None,
        use_kv_cache=True,
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
            use_kv_cache:   是否使用 KV Cache
        """
        B = idx.size(0)
        device = idx.device
        dtype = next(self.parameters()).dtype
        
        # 支持多个 EOS token
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)
        
        # 跟踪哪些序列已经结束
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        if use_kv_cache:
            total_len = idx.size(1) + max_new_tokens
            kv_cache = self._create_kv_cache(B, total_len, device, dtype)

            # Prefill
            logits, _  = self.forward(idx, kv_cache=kv_cache)
            # logits shape: (B, T, vocab_size)

            # 收集生成的 token
            generated = idx

            for _ in range(max_new_tokens):
                # 如果所有序列都结束了，提前退出
                if finished.all():
                    break

                next_logits = logits[:, -1, :]

                # 对已结束序列填 pad
                if finished.any() and pad_token_id is not None:
                    next_logits[finished] = -float('Inf')
                    next_logits[finished, pad_token_id] = 0

                # 采样（传入已生成的 token 用于重复惩罚）
                idx_next = self._sample(next_logits, temperature, top_k, top_p, do_sample, generated=generated)

                generated = torch.cat((generated, idx_next), dim=1)

                # 检查 EOS
                if eos_token_id is not None:
                    is_eos = (idx_next.squeeze(1).unsqueeze(1) == eos_token_id.unsqueeze(0)).any(dim=1)
                    finished = finished | is_eos

                # 如果所有序列都结束了，不再执行 forward
                if finished.all():
                    break

                # Decode step: 只送入新 token（为下一次迭代准备 logits）
                logits, _ = self.forward(idx_next, kv_cache=kv_cache)

            return generated

        else:
            # 原始实现, 兼容旧代码
            for _ in range(max_new_tokens):
                if finished.all():
                    break
                    
                idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                
                if finished.any() and pad_token_id is not None:
                    logits[finished] = -float('Inf')
                    logits[finished, pad_token_id] = 0
                
                idx_next = self._sample(logits, temperature, top_k, top_p, do_sample, generated=idx, repetition_penalty=repetition_penalty)
                idx = torch.cat((idx, idx_next), dim=1)
                
                if eos_token_id is not None:
                    is_eos = (idx_next.squeeze(1).unsqueeze(1) == eos_token_id.unsqueeze(0)).any(dim=1)
                    finished = finished | is_eos

            return idx
        

    def _sample(self, logits, temperature, top_k, top_p, do_sample, generated=None, repetition_penalty=1.1):
        """采样辅助函数, 从 logits 中采样下一个 token
        
        Args:
            logits: (B, vocab_size) 的 logits
            temperature: 温度参数
            top_k: top-k 采样
            top_p: nucleus 采样
            do_sample: 是否采样
            generated: (B, seq_len) 已生成的 token ids，用于重复惩罚
            repetition_penalty: 重复惩罚系数，默认 1.5（>1 时降低已生成 token 的概率）
        """
        # 应用重复惩罚（参考 HuggingFace RepetitionPenaltyLogitsProcessor）
        if generated is not None and repetition_penalty > 1.0:
            for i in range(logits.size(0)):
                unique_tokens = torch.unique(generated[i])
                # 过滤掉特殊 token，不惩罚 stop tokens
                mask = torch.ones(len(unique_tokens), dtype=torch.bool, device=unique_tokens.device)
                for sid in [151643, 151645, 151644]:  # endoftext, im_end, im_start
                    mask &= (unique_tokens != sid)
                penalize_tokens = unique_tokens[mask]
                # 正 logits 除以 penalty，负 logits 乘以 penalty
                scores = logits[i, penalize_tokens]
                logits[i, penalize_tokens] = torch.where(
                    scores > 0, scores / repetition_penalty, scores * repetition_penalty
                )
        
        if do_sample and temperature > 0:
            logits = logits / temperature

        if do_sample and top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if do_sample and top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')

        if do_sample:
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        return idx_next
    
    @classmethod
    def from_pretrained(cls, model_name="Qwen/Qwen2.5-7B", load_weights=True):
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
        
        Args:
            model_name: HF 模型名称或路径
            load_weights: 是否加载 HF 预训练权重 (默认 True)。设为 False 时只创建模型结构。
        """
        from transformers import AutoModelForCausalLM, AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # rope_base: 兼容不同版本的 HF config 属性名
        rope_base = getattr(hf_config, 'rope_theta', None) \
                    or getattr(hf_config, 'rope_base', None) \
                    or 1000000.0

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

        # 创建我们的模型（随机初始化）
        model = cls(config)

        if not load_weights:
            # 仅创建模型结构，不加载 HF 权重（用于断点续训）
            return model

        # 加载 HF 权重
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True,
        )
        hf_sd = hf_model.state_dict()

        # 键名映射
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